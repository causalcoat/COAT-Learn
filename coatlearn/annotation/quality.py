"""
Factor Quality Gates

提供用于 COAT-MB 的因子质量检查门控函数。
这些函数用于过滤不符合质量标准的因子。
"""

import numpy as np
from typing import List, Callable
from causallearn.utils.cit import CIT
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def create_TY_independence_gate(
    T_values: np.ndarray, Y_values: np.ndarray, alpha: float = 0.01
) -> Callable[[np.ndarray], bool]:
    """
    旧启发式对照：给定候选因子后若T和Y条件独立则拒绝。

    这不是因子有效性或调整准则：有效混杂变量也可能使T/Y条件独立。
    当前ADJ默认不使用此额外门控；保留函数仅用于显式旧逻辑对照。

    Args:
        T_values: 处理变量 (Treatment) 的值
        Y_values: 结果变量 (Outcome) 的值
        alpha: 显著性水平阈值（默认0.01）

    Returns:
        门控函数，接受因子值数组，返回是否通过检查
    """
    _TY = np.column_stack([T_values, Y_values])

    def gate(factor_values: np.ndarray) -> bool:
        """检查因子是否与 T-Y 条件独立"""
        _data = np.column_stack([_TY, factor_values])
        _data = np.nan_to_num(_data, 0)
        kci_obj = CIT(_data, "kci")

        # 测试在条件化因子后，T和Y是否独立
        pValue = kci_obj(0, 1, [2])

        if pValue > alpha:
            return False

        return True

    return gate


def create_collinearity_gate(
    Y_values: np.ndarray,
    T_values: np.ndarray = None,
    r2_threshold: float = 0.5,
) -> Callable[[np.ndarray], bool]:
    """
    创建多重共线性检查门控

    检查因子是否被 T 和 Y（或仅 Y）过度线性预测（多重共线性）。
    如果 R² >= threshold，说明因子与参考变量高度共线，应被拒绝。

    Args:
        Y_values: 结果变量 (Outcome) 的值
        T_values: 处理变量 (Treatment) 的值，可选。如果提供，则与 Y 一起检查共线性
        r2_threshold: R² 阈值（默认0.5）

    Returns:
        门控函数，接受因子值数组，返回是否通过检查
    """
    if T_values is not None:
        _X = np.column_stack([T_values, Y_values])
    else:
        _X = Y_values.reshape(-1, 1) if Y_values.ndim == 1 else Y_values

    def gate(factor_values: np.ndarray) -> bool:
        """检查因子是否与 T-Y 存在多重共线性"""
        _y = factor_values

        # 处理 NaN
        mask = ~np.isnan(_y)
        if mask.sum() < 3:
            return False

        _X_valid = _X[mask]
        _y_valid = _y[mask]

        # 创建并训练线性回归模型
        model = LinearRegression()
        model.fit(_X_valid, _y_valid)

        # 预测
        y_pred = model.predict(_X_valid)

        # 计算 R² 分数
        r2 = r2_score(_y_valid, y_pred)

        if r2 >= r2_threshold:
            return False

        return True

    return gate


def create_TY_quality_gate_list(
    T_values: np.ndarray,
    Y_values: np.ndarray,
    alpha: float = 0.01,
    r2_threshold: float = 0.5,
) -> List[Callable[[np.ndarray], bool]]:
    """
    创建 T-Y 场景下的完整质量门控列表

    组合了独立性检查和多重共线性检查两个门控。

    Args:
        T_values: 处理变量 (Treatment) 的值
        Y_values: 结果变量 (Outcome) 的值
        alpha: 独立性检验的显著性水平（默认0.01）
        r2_threshold: 共线性检查的 R² 阈值（默认0.5）

    Returns:
        门控函数列表
    """
    return [
        create_TY_independence_gate(T_values, Y_values, alpha),
        create_collinearity_gate(T_values, Y_values, r2_threshold),
    ]
