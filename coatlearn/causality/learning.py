"""Graph construction and collinearity selection used by COAT-ADJ."""
import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from .statistics import vif_or_inf
from .graph import graph_wrapper

class GraphUtils:
    """
    图生成和处理工具类
    """

    @staticmethod
    def clean_nodes_vif(
        data_interface: pd.DataFrame,
        W_list: List[str],
        T_name: str,
        Y_name: str,
        max_vif: float = 5.0,
    ) -> List[str]:
        """
        使用 VIF 检测并移除多重共线性因子

        Args:
            data_interface: 数据接口 DataFrame
            W_list: 节点列表
            T_name: 处理变量名（始终保留）
            Y_name: 结果变量名（始终保留）
            max_vif: 最大 VIF 阈值

        Returns:
            清理后的节点列表
        """
        W_list = list(set(W_list))
        hold_features = [col for col in [T_name, Y_name] if col in W_list]
        features = [col for col in W_list if col not in hold_features]

        if len(features) == 0:
            return W_list.copy()

        selected_features = W_list.copy()

        while len(selected_features) > len(hold_features):
            check_features = [f for f in selected_features if f not in hold_features]
            if not check_features:
                break

            # 每次循环重新构建当前特征的数据矩阵；缺失值不被当成 0 参与 VIF
            X_current = data_interface[selected_features].values
            vif_values = {
                feature: vif_or_inf(X_current, selected_features.index(feature))
                for feature in check_features
            }

            max_vif_value = max(vif_values.values()) if vif_values else 0
            if max_vif_value <= max_vif:
                break

            # 移除 VIF 最高的特征
            highest_vif_feature = max(vif_values.items(), key=lambda x: x[1])[0]
            selected_features.remove(highest_vif_feature)
            logging.info(
                f"Removed {highest_vif_feature} with VIF = {max_vif_value:.2f}"
            )

        # 【修复】保持顺序，不使用 set()
        result_features = [f for f in selected_features if f in hold_features]
        result_features += [f for f in selected_features if f not in hold_features]

        return result_features

    @staticmethod
    def generate_combined_graph(
        data_interface: pd.DataFrame,
        W_list: List[str],
        alpha: float,
        test_method: str,
        T_idx: Optional[int] = None,
        Y_idx: Optional[int] = None,
        use_lingam: bool = False,
        fci_depth: int = 1,
        assume_treatment_causes_outcome: bool = False,
        prior_diagnostics: Optional[Dict[str, Any]] = None,
        assume_factors_precede_treatment: bool = False,
    ):
        """
        生成因果图（FCI 或 LiNGAM）

        约束为显式 opt-in：默认 False 时 FCI 无背景知识（bk=None）。
        True 表示已知直接边 T->Y，保护邻接并定向；这是研究者先验，
        不是 CI 证据或算法发现。调用方必须记录 direction_constraints。

        Args:
            data_interface: 数据接口 DataFrame
            W_list: 变量列表
            alpha: 显著性水平
            test_method: 独立性测试方法
            T_idx: 处理变量索引（仅在 assume_treatment_causes_outcome=True 时必需）
            Y_idx: 结果变量索引（同上）
            use_lingam: 是否使用 LiNGAM
            fci_depth: FCI 算法深度
            assume_treatment_causes_outcome: 显式 required 邻接和方向先验
            prior_diagnostics: 可选审计输出；记录先验保留而非统计 p 值

        Returns:
            FCI: (g, edges, bk)——未开启方向假设时 bk 为 None；LiNGAM: g
        """
        from causallearn.search.ConstraintBased.FCI import fci

        annoted_values = data_interface[W_list].values

        # LiNGAM
        if use_lingam:
            import lingam
            annoted_values = np.nan_to_num(annoted_values, nan=0.0)

            lingam_model = lingam.DirectLiNGAM()
            lingam_model.fit(annoted_values)

            lingam_graph = np.zeros_like(lingam_model.adjacency_matrix_)
            for i in range(lingam_graph.shape[0]):
                for j in range(lingam_graph.shape[1]):
                    if lingam_model.adjacency_matrix_[i, j] != 0:  # j --> i
                        lingam_graph[i, j] = 1
                        lingam_graph[j, i] = -1

            g = graph_wrapper(graph=lingam_graph, labels=W_list)
            return g

        # FCI
        if test_method not in ["mv_fisherz", "chisq", "gsq"]:
            annoted_values = np.nan_to_num(annoted_values, nan=0.0)

        # Background knowledge alone only orients surviving edges in the
        # installed FCI. Protect the explicit required pair during CI deletion.
        bk = None
        if assume_treatment_causes_outcome:
            if T_idx is None or Y_idx is None:
                raise ValueError(
                    "T_idx and Y_idx must be specified when "
                    "assume_treatment_causes_outcome=True"
                )
            from coatlearn.causality.priors import fci_with_required_edge
            tiers = None
            if assume_factors_precede_treatment:
                # Every measured factor precedes T, which precedes Y.
                tiers = [0] * annoted_values.shape[1]
                tiers[T_idx] = 1
                tiers[Y_idx] = 2
            return fci_with_required_edge(
                annoted_values, T_idx, Y_idx, alpha=alpha, test_method=test_method,
                depth=fci_depth, fci_func=fci, diagnostics=prior_diagnostics,
                tiers=tiers)

        # 默认路径完全沿用无先验 FCI。
        g, edges = fci(
            annoted_values,
            alpha=alpha,
            depth=fci_depth,
            independence_test_method=test_method,
            verbose=False,
            background_knowledge=bk,
        )

        return (g, edges, bk)

