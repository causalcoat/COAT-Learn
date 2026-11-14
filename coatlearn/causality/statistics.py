"""Collinearity and IPW balance calculations, without model calls."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

def vif_or_inf(values, column: int) -> float:
    """VIF of one column computed on complete rows only.

    Factors that keep missing cells (general template) would otherwise make
    statsmodels raise or return nan, which callers silently treat as "infinitely
    collinear" and delete the factor.  Without missing cells this is exactly the
    plain VIF call.
    """
    matrix = np.asarray(values, dtype=float)
    finite_rows = np.isfinite(matrix).all(axis=1)
    if finite_rows.sum() <= matrix.shape[1] + 1:
        return float("inf")
    try:
        value = variance_inflation_factor(matrix[finite_rows], column)
    except Exception:
        return float("inf")
    return float("inf") if np.isnan(value) else float(value)


def is_ipw_valid(
    X,
    T,
    Y,
    ps_model=None,
    auc_thresh=0.6,
    smd_thresh=0.15,
    extreme_ps_thresh=0.05,
    random_state=None,
):
    """IPW 协变量平衡诊断，返回 ``(通过与否, 原因)``。

    当前仅在最大加权均值标准化差异（SMD）超过 ``smd_thresh`` 时返回
    ``(False, "SMD too high")``，否则返回 ``(True, "")``。
    AUC 和极端倾向得分比例仍会计算，但对应的拒绝分支未启用；
    ``auc_thresh``、``extreme_ps_thresh`` 和 ``Y`` 当前不参与判定。
    通过此诊断不证明调整集有效、效应可识别或估计无偏。
    """

    # 初始化默认模型
    if ps_model is None:
        ps_model = HistGradientBoostingClassifier(
            max_iter=100, random_state=random_state
        )

    # # 1. 检查AUC
    ps = cross_val_predict(ps_model, X, T, method="predict_proba", cv=5)[:, 1]
    auc = roc_auc_score(T, ps)
    # if auc < auc_thresh:
    #     return False, "AUC too low"

    # # 2. 检查极端倾向得分
    extreme_ratio = np.mean((ps < 0.1) | (ps > 0.9))
    # if extreme_ratio > extreme_ps_thresh:
    #     return False, "Extreme propensity score ratio too high"

    # 3. 检查SMD平衡性
    weights = (T / ps) + ((1 - T) / (1 - ps))
    smd_values = []
    for i in range(X.shape[1]):
        treated = X[T == 1, i]
        control = X[T == 0, i]
        w_treated = weights[T == 1]
        w_control = weights[T == 0]

        mean_diff = np.average(treated, weights=w_treated) - np.average(
            control, weights=w_control
        )
        pooled_std = np.sqrt((np.var(treated) + np.var(control)) / 2)
        smd = abs(mean_diff / pooled_std)
        smd_values.append(smd)

    if max(smd_values) > smd_thresh:
        return False, "SMD too high"

    return True, ""  # 所有检查通过

