"""
COAT-PAG: COAT Partial Ancestral Graph Discovery Algorithm

该模块实现了从非结构化文本中发现因果因子并构建 PAG 的算法。
基于 coat_mb.py，添加了多目标变量支持和 PAG 特定功能。
"""

import sys
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from typing import Dict, Set, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.cit import CIT

from coatlearn.discovery.mb import COATMB, default_annotation_batching
from coatlearn.causality.graph import get_bk
from coatlearn.utils.logging import Logger
from coatlearn.causality.statistics import vif_or_inf


class COATPAG:
    """
    COAT PAG 发现算法（基于 COATMB）

    功能：
    - 从文本数据中发现多个变量的因果因子
    - 构建 PAG（部分祖先图）
    - 支持多目标变量（U_list）
    - 实现多阶段算法（MB发现 → 不确定节点扩展 → 因子池检索 → 图生成）
    """

    def __init__(
        self,
        llm_proposer,
        llm_annotator,
        U_list: List[str],
        init_kv: Dict[str, List],
        text_values: np.ndarray,
        log_root_path: Optional[str] = None,
    ):
        """
        初始化 COAT-PAG 算法

        Args:
            llm_proposer: LLM 提议器
            llm_annotator: LLM 标注器
            U_list: 已知变量列表（如 ['U_1', 'U_2', ...]）
            init_kv: 初始变量字典 {U_1: [...], U_2: [...], ...}
            text_values: 文本数据
            log_root_path: 日志根目录
        """
        self.llm_proposer = llm_proposer
        self.llm_annotator = llm_annotator
        self.U_list = U_list
        self.init_kv = init_kv
        self.text_values = text_values

        # 创建 meta_df（用于 prompt 生成）
        self.meta_df = pd.DataFrame(init_kv)
        self.meta_df["X"] = text_values

        # 算法状态
        self.annotation = deepcopy(init_kv)
        self.proposed_factors_history = []
        self.factor_pools = {}
        self._mb_by_target: Dict[str, COATMB] = {}
        self.factor_prediction_filter = None  # supplied only by an ADJ parent

        # 错误登记（由各 COATMB 子实例汇总而来，供 facade 检查）
        self.annotation_errors = {}  # {(factor_name, sample_idx): "失败原因"}
        self.proposal_errors = {}
        self.judge_errors = {}

        # 迭代计数
        self.round_idx = 0

        # 配置参数（默认值）
        self.config = {
            "num_workers": 50,
            "vif_threshold": 5.0,
            "alpha": 0.05,
            "test_method": "rcit",
            "fci_depth": 2,
            "prompt_sample_size": 5,
            "save_llm_conversation": False,
            "save_factor_audit": True,
            "factor_judge": None,
            "factor_dedup": None,
            "annotation_batching": default_annotation_batching(llm_annotator),
        }

        # 设置日志
        if log_root_path is None:
            log_root_path = (
                f"./exp_log_folders/log_coat_pag_{time.strftime('%Y%m%d_%H%M%S')}"
            )
        self.log_root_path = log_root_path

        import os

        os.makedirs(log_root_path, exist_ok=True)
        # Logger 只做文件写入（straight_write）；库内不劫持 sys.stdout/stderr，
        # 也不重配 root logger。需要控制台/文件日志时由调用者配置 logging，
        # 或显式调用 self.logger.attach_root_file_logging(...)（opt-in）。
        self.logger = Logger(log_root_path)

    def set_config(self, **kwargs):
        """更新配置参数；记录 opt-in judge 的实际判据，不触发请求。"""
        if "factor_judge" in kwargs and kwargs["factor_judge"] is not None:
            from coatlearn.annotation.judge import configured_judge

            template = kwargs.get("factor_template") or self.config.get("factor_template", "simple")
            self.config["factor_judge_policy"] = configured_judge(kwargs["factor_judge"], template).policy()
        elif "factor_judge" in kwargs:
            self.config["factor_judge_policy"] = None
        if "factor_dedup" in kwargs:
            from coatlearn.annotation.dedup import configured_deduplicator, core_dedup_policy

            comparator, context = configured_deduplicator(kwargs["factor_dedup"])
            self.config["factor_dedup_policy"] = core_dedup_policy(comparator, context)
        self.config.update(kwargs)

    # ========================================================================
    # 辅助方法：针对每个目标变量创建 COATMB 实例
    # ========================================================================

    def _create_coat_mb_for_target(self, target_name: str) -> COATMB:
        """为单个目标变量创建 COATMB 实例"""
        # 创建包含目标变量的 meta_df
        target_vars = self.U_list + [
            f for f in self.annotation.keys() if f not in self.U_list
        ]

        # 构建该目标变量的 meta_df
        target_meta_df = pd.DataFrame({k: self.annotation[k] for k in target_vars})
        if "X" not in target_meta_df.columns:
            target_meta_df["X"] = self.text_values

        # 创建 COATMB 实例
        coat_mb = COATMB(
            llm_proposer=self.llm_proposer,
            llm_annotator=self.llm_annotator,
            target_name=target_name,
            meta_df=target_meta_df,
            text_col="X",
        )

        # 继承 annotation（已有标注）
        coat_mb.annotation = deepcopy(self.annotation)
        coat_mb.factor_pools = deepcopy(self.factor_pools)
        coat_mb.proposed_factors_history = deepcopy(self.proposed_factors_history)
        coat_mb.iteration = self.round_idx
        coat_mb.factor_prediction_filter = self.factor_prediction_filter

        # 设置配置
        coat_mb.set_config(**self.config)

        return coat_mb

    # ========================================================================
    # Prompt 方法（使用 mb/stress_test.py 的模板）
    # ========================================================================

    def generate_proposal_prompt(
        self,
        target_name: str,
        chosen_indices: np.ndarray,
        existing_factors: Set[str] = None,
    ) -> str:
        """
        生成 LLM 因子提议 prompt

        Args:
            target_name: 目标变量名称
            chosen_indices: 被选中的样本索引
            existing_factors: 已存在的因子

        Returns:
            格式化的 prompt
        """
        if existing_factors is None:
            existing_factors = set(self.annotation.keys())

        # 按目标变量值分组
        data = ""

        # 获取目标变量的值
        if target_name in self.meta_df.columns:
            target_values = self.meta_df[target_name].values
        else:
            target_values = self.annotation[target_name]

        for f_v in np.sort(list(set(target_values[chosen_indices]))):
            data += f"\n\n## Group with '{target_name}' = {f_v}\n"

            # 找到这些值对应的索引
            these_indices = np.arange(len(chosen_indices))[
                (target_values[chosen_indices] == f_v)
            ]

            # 每组最多选 3 个样本
            if len(these_indices) > 3:
                these_indices = np.random.choice(these_indices, 3, replace=False)

            for idx in these_indices:
                actual_idx = chosen_indices[idx]
                text = self.meta_df.iloc[actual_idx]["X"]

                # 清理文本
                if isinstance(text, str):
                    text = text.strip()
                else:
                    text = str(text)

                data += f"\n- {text}"

        prompt = f"""
You are an excellently helpful AI assistant for analyzing, abstracting, and processing data. Now try your best.

# Data

{data}

# Tasks: Factor Abstraction.

**What are the high-level factors behind the text that contribute to the allocation of groups?**

- Propose your candidate factors based on your observation on given data.
- The proposed factors should be diverse and different semantically. Their overlapping should be minimized.

# About Output

Your output should contain parts described as follows.

**Part 1**: Consideration.

In this part, feel free to write down the process of your considerations.
Hint: You need to abstract, identify, and design suitable factors with corresponding criteria, and each factor is only allowed to take value from [-1, 0, 1].

**Part 2**: Final Output.

In this part, you are required to report the factors that you decided to use.
- Each factors' value should be from **[-1, 0, 1]**
- **You should give specific criterion for each value of each factor.**
- Keep the criterion for value 0 fixed as "Otherwise; or not mentioned".

Report the factors **in following template:**

```
**Factor Name**
- 1: [Positive Criterion]
- 0: [Otherwise; or not mentioned].
- -1: [Negative Criterion]
```
"""
        return prompt

    def prompt_factor_annotation(self, factor_state: str, text: str) -> str:
        """
        生成因子标注 prompt

        Args:
            factor_state: 因子定义
            text: 待标注的文本

        Returns:
            格式化的 prompt
        """
        prompt = f"""

You are an expert data annotation assistant. Your task is to extract or calculate strictly numeric values from text.

# Data

```
{text.strip()}
```

# Annotation Task
Evaluate the value of this factor on the given data.

```
{factor_state.strip()}
```

Examples:
Good: "The value is: 3.2"
Good: "The value is: -1"
Good: "The value is: 0"
Bad: "The value is: €3.2"
Bad: "The value is: -1K"

"""
        return prompt

    # ========================================================================
    # 核心算法流程
    # ========================================================================

    def singleRoundCOAT(
        self,
        round_idx: int,
        this_target_name: str,
        test_method: str,
        alpha: float,
        enforce_vif_check_names: List[str] = None,
        just_from_pools: bool = False,
    ) -> Set[str]:
        """
        执行单轮 COAT 迭代

        Args:
            round_idx: 轮次索引
            this_target_name: 目标变量名称
            test_method: 独立性测试方法
            alpha: 显著性水平
            enforce_vif_check_names: VIF 检查的变量列表
            just_from_pools: 是否仅从因子池检索

        Returns:
            更新后的 V（包含目标和发现的因子）
        """
        self.round_idx = round_idx

        logging.info(f"\n{'=' * 60}")
        logging.info(f"Round {round_idx}: Target = {this_target_name}")
        logging.info(f"{'=' * 60}")

        # 每个目标独立维护当前 MB；仅第一次调用初始化。后续 proposal 的
        # 聚类/反馈使用上一次估计，但 CI 仍可重新计算并覆盖它。
        coat_mb = self._mb_by_target.get(this_target_name)
        if coat_mb is None:
            coat_mb = self._create_coat_mb_for_target(this_target_name)
            coat_mb.V = {this_target_name}
            self._mb_by_target[this_target_name] = coat_mb
        else:
            # 同步其他目标新增的测量，不继承其他目标的 MB。
            coat_mb.annotation = deepcopy(self.annotation)
            coat_mb.factor_pools = deepcopy(self.factor_pools)
            coat_mb.proposed_factors_history = deepcopy(self.proposed_factors_history)
            coat_mb.set_config(**self.config)
        coat_mb.iteration = round_idx - 1  # single_round 自增；全局审计编号不冲突
        history_start = len(coat_mb.proposed_factors_history)

        output_dir = Path(self.log_root_path) if self.log_root_path else None
        new_V = coat_mb.single_round(
            output_dir=output_dir,
            alpha=alpha,
            test_method=test_method,
            just_from_pools=just_from_pools,
        )

        for factor, values in coat_mb.annotation.items():
            if factor not in self.annotation:
                self.annotation[factor] = values
        for factor, description in coat_mb.factor_pools.items():
            if factor not in self.factor_pools:
                self.factor_pools[factor] = description

        # 只合并本轮增量；继承的历史不能再追加一遍。
        self.proposed_factors_history.extend(
            coat_mb.proposed_factors_history[history_start:]
        )
        self.annotation_errors.update(coat_mb.annotation_errors)
        self.proposal_errors.update(coat_mb.proposal_errors)
        self.judge_errors.update(coat_mb.judge_errors)

        return set(new_V)

    def _discover_target_mb(
        self, target_name: str, num_round: int, test_method: str, alpha: float,
    ) -> Set[str]:
        """Algorithm 1：连续更新同一目标的 MB，稳定或达到轮数上限时返回。"""
        mb = self._mb_by_target.get(target_name)
        previous = set(mb.V) if mb is not None else {target_name}
        for _ in range(num_round):
            self.round_idx += 1
            current = self.singleRoundCOAT(
                self.round_idx, target_name, test_method, alpha,
            )
            if current == previous:
                logging.info(f"  {target_name}: MB 未变化，结束本次搜索")
                return current
            previous = set(current)
        return previous

    # ========================================================================
    # PAG 特定功能
    # ========================================================================

    def generate_combined_graph(
        self,
        W_list: List[str],
        alpha: float,
        test_method: str,
        T_idx: Optional[int] = None,
        Y_idx: Optional[int] = None,
    ):
        """
        使用 FCI 算法生成因果图

        Args:
            W_list: 变量列表（同时定义图矩阵的节点顺序）
            alpha: 显著性水平
            test_method: 独立性测试方法
            T_idx: 处理变量索引（可选）。只有当调用者显式给出 T_idx/Y_idx 时
                才构建方向背景知识；默认 None，不把 U_list[0]/U_list[1]
                （或任何按索引的位置）强制为处理/结果方向。
            Y_idx: 结果变量索引（可选）

        Returns:
            (g, edges, bk): 因果图、边、背景知识（未指定 T_idx/Y_idx 时 bk 为 None）
        """
        data_interface = pd.DataFrame(self.annotation)
        annoted_values = data_interface[W_list].values.copy()

        # 处理缺失值
        if test_method not in ["mv_fisherz", "chisq", "gsq"]:
            annoted_values = np.nan_to_num(annoted_values, nan=0.0)

        # ADJ may pass a known edge by NAME; never infer it from U_list order.
        required_edge = self.config.get('required_causal_edge')
        if required_edge is not None:
            from coatlearn.causality.priors import fci_with_required_edge
            source, target = required_edge
            self._required_edge_ci = {}
            tiers = None
            if self.config.get("assume_factors_precede_treatment", False):
                source_idx, target_idx = W_list.index(source), W_list.index(target)
                tiers = [0] * len(W_list)
                tiers[source_idx] = 1
                tiers[target_idx] = 2
            return fci_with_required_edge(
                annoted_values, W_list.index(source), W_list.index(target),
                alpha=alpha, test_method=test_method, depth=self.config['fci_depth'],
                fci_func=fci, diagnostics=self._required_edge_ci, tiers=tiers)

        # 默认不施加任何方向约束；仅显式提供 T_idx/Y_idx 时才构建背景知识。
        # 注意：get_bk 当前不添加任何约束（见 causality.graph.get_bk），显式传入也只得到空 bk。
        if T_idx is not None and Y_idx is not None:
            bk = get_bk(T_idx, Y_idx, annoted_values.shape[1])
        else:
            bk = None

        # 运行 FCI 算法
        g, edges = fci(
            annoted_values,
            alpha=alpha,
            depth=self.config["fci_depth"],
            independence_test_method=test_method,
            verbose=False,
            background_knowledge=bk,
        )

        return g, edges, bk

    def identify_uncertain_nodes(self, g, W_list: List[str]) -> Set[str]:
        """
        识别 PAG 中的不确定节点（含 circle 端点的边）

        Args:
            g: 因果图
            W_list: 变量列表

        Returns:
            不确定节点的集合
        """
        uncertain_nodes_idx = np.where((g.graph == 2).any(1))[0]
        uncertain_nodes = set([W_list[idx] for idx in uncertain_nodes_idx])
        return uncertain_nodes

    def clean_nodes_vif(
        self, node_list: List[str], preserve_list: List[str], max_vif: float
    ) -> List[str]:
        """
        使用 VIF 检测并移除多重共线性因子

        Args:
            node_list: 节点列表
            preserve_list: 始终保留的变量列表
            max_vif: 最大 VIF 阈值

        Returns:
            清理后的节点列表
        """
        if self.factor_prediction_filter is not None:
            node_list = self.factor_prediction_filter.filter(node_list, self.annotation, self.factor_pools)
            preserve_list = self.factor_prediction_filter.filter(preserve_list, self.annotation, self.factor_pools)
        new_list = list(preserve_list)

        for each_factor in node_list:
            if each_factor in new_list:
                continue

            try:
                data_interface = pd.DataFrame(self.annotation)
                this_vif = vif_or_inf(
                    data_interface[new_list + [each_factor]].values, len(new_list)
                )

                if this_vif < max_vif:
                    new_list.append(each_factor)
                else:
                    logging.info(f"  拒绝因子 {each_factor}, VIF={this_vif:.2f}")
            except Exception as e:
                logging.info(f"  因子 {each_factor} VIF 计算失败: {e}")

        return new_list

    # ========================================================================
    # 完整的多阶段算法
    # ========================================================================

    def run_algorithm(
        self, num_round: int = 2, output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        运行完整的 COAT-PAG 算法

        Args:
            num_round: 每个初始/扩展目标的 MB 搜索轮数上限（MB 不变则提前停止）；
                同时保留原有 PAG 扩展批次上限，每批最多处理 3 个不确定节点。
            output_dir: 输出目录

        Returns:
            结果字典，包含：
            - U_list: 初始变量列表
            - stages: 每个阶段的状态
            - final_V: 最终因子列表
            - llm_proposed_count: LLM 提议因子总数
            - llm_proposed_factors: 所有提议的因子
        """
        result = {
            "U_list": self.U_list,
            "stages": [],
            "final_V": None,
            "llm_proposed_count": 0,
            "llm_proposed_factors": [],
        }

        test_method = self.config["test_method"]
        alpha = self.config["alpha"]

        # ========================================================================
        # Stage 1: Markov Blanket 发现
        # ========================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Stage 1: Markov Blanket Discovery")
        logging.info("=" * 80)

        V_list = self.U_list.copy()

        MB_results = {}
        for u_name in self.U_list:
            # Algorithm 2 调用完整的有界 Algorithm 1，而非互不连续的单轮。
            u_MB = self._discover_target_mb(u_name, num_round, test_method, alpha)
            MB_results[u_name] = u_MB

            proposed_this_round = list(set(u_MB) - set(self.U_list))
            result["llm_proposed_factors"].extend(proposed_this_round)
            result["llm_proposed_count"] += len(proposed_this_round)
            logging.info(f"[DEBUG] {u_name} MB search completed")
            time.sleep(5)

        # 清理因子列表
        logging.info(f"\n[DEBUG] Starting Stage 1 cleanup...")
        logging.info(f"[DEBUG] MB_results: {MB_results}")
        logging.info(f"[DEBUG] self.U_list: {self.U_list}")

        try:
            combined_MB = set()
            for mb in MB_results.values():
                combined_MB |= set(mb)
            logging.info(f"[DEBUG] combined_MB: {combined_MB}")

            new_factors = combined_MB - set(self.U_list)
            logging.info(f"[DEBUG] new_factors: {new_factors}")

            V_list = self.U_list + list(new_factors)
            logging.info(f"[DEBUG] V_list constructed: {V_list}")
        except Exception as e:
            logging.error(f"[ERROR] Failed to construct V_list: {e}")
            import traceback

            logging.error(traceback.format_exc())
            raise
        V_list = self.clean_nodes_vif(V_list, self.U_list, self.config["vif_threshold"])
        logging.info(f"[DEBUG] After VIF cleaning: {V_list}")

        logging.info(f"Stage 1 V_list: {V_list}")

        # 记录 Stage 1 状态
        result["stages"].append(
            {
                "stage": "Stage 1: Markov Blanket Discovery",
                "V_size": len(V_list),
                "V_list": V_list,
                "new_factors": list(set(V_list) - set(self.U_list)),
            }
        )

        # 生成初始因果图（不指定 T_idx/Y_idx：不对 U_list 施加处理/结果方向约束）
        logging.info(f"\n[DEBUG] Generating initial causal graph...")
        try:
            g, edges, bk = self.generate_combined_graph(V_list, alpha, test_method)
            logging.info(f"[DEBUG] Initial causal graph generated successfully")
        except Exception as e:
            logging.error(f"[ERROR] Failed to generate initial causal graph: {e}")
            raise
        S_mark = set(self.U_list)
        logging.info(f"[DEBUG] S_mark initialized: {S_mark}")

        # ========================================================================
        # Stage 2-N: 扩展不确定节点
        # ========================================================================
        logging.info(f"\n[DEBUG] About to start Stage 2-N loop (num_round={num_round})")
        for round_iter in range(num_round):
            logging.info("\n" + "=" * 80)
            logging.info(f"Stage {round_iter + 2}: Extending Undetermined Nodes")
            logging.info("=" * 80)

            # 生成当前因果图（同上，无方向约束）
            g, edges, bk = self.generate_combined_graph(V_list, alpha, test_method)
            time.sleep(5)

            # 识别不确定节点
            uncertain_nodes = self.identify_uncertain_nodes(g, V_list)
            to_extend = uncertain_nodes - S_mark

            if len(to_extend) > 3:
                to_extend = set(np.random.choice(list(to_extend), 3, replace=False))

            logging.info(f"Extending undetermined nodes: {', '.join(to_extend)}")

            # 扩展不确定节点
            for each_target in to_extend:
                extended_MB = self._discover_target_mb(
                    each_target, num_round, test_method, alpha,
                )

                # 记录
                proposed_this_round = list(set(extended_MB) - set(V_list))
                result["llm_proposed_factors"].extend(proposed_this_round)
                result["llm_proposed_count"] += len(proposed_this_round)

                V_list = list(set(V_list + list(extended_MB)))
                V_list = self.clean_nodes_vif(
                    V_list, self.U_list, self.config["vif_threshold"]
                )
                S_mark.add(each_target)

                time.sleep(1)

            # 记录 Stage 2-N 状态
            result["stages"].append(
                {
                    "stage": f"Stage {round_iter + 2}: Extending Undetermined Nodes",
                    "V_size": len(V_list),
                    "V_list": V_list,
                    "new_factors": list(set(V_list) - set(self.U_list)),
                    "extended_nodes": list(to_extend),
                }
            )

        # ========================================================================
        # Stage N+1: 从因子池检索
        # ========================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Stage N+1: Including More Nodes from Pools")
        logging.info("=" * 80)

        V_list_old = deepcopy(V_list)

        for each_target in V_list:
            if each_target not in S_mark:
                self.round_idx += 1
                pool_MB = self.singleRoundCOAT(
                    self.round_idx,
                    this_target_name=each_target,
                    test_method=test_method,
                    alpha=alpha,
                    enforce_vif_check_names=V_list,
                    just_from_pools=True,
                )

                V_list = list(set(V_list + list(pool_MB)))
                S_mark.add(each_target)

                time.sleep(3)
                break  # 仅执行一轮

        V_list = self.clean_nodes_vif(V_list, self.U_list, self.config["vif_threshold"])

        # 记录 Stage N+1 状态
        result["stages"].append(
            {
                "stage": "Stage N+1: Including More Nodes from Pools",
                "V_size_before": len(V_list_old),
                "V_size_after": len(V_list),
                "V_list": V_list,
                "new_factors": list(set(V_list) - set(self.U_list)),
            }
        )

        # ========================================================================
        # Stage N+2: 生成最终因果图
        # ========================================================================
        logging.info("\n" + "=" * 80)
        logging.info("Stage N+2: Generate Final Causal Graph")
        logging.info("=" * 80)

        # 最终 VIF 过滤
        data_interface = pd.DataFrame(self.annotation)
        _V_list = deepcopy(V_list)
        V_list = list(self.U_list)

        for each_factor in _V_list:
            if each_factor in V_list:
                continue

            try:
                this_vif = vif_or_inf(
                    data_interface[V_list + [each_factor]].values, len(V_list)
                )
                logging.info(f"VIF for {each_factor}: {this_vif}")

                if this_vif < self.config["vif_threshold"]:
                    V_list.append(each_factor)
            except Exception as e:
                logging.info(f"VIF 计算失败 for {each_factor}: {e}")

        # 生成最终因果图（不指定 T_idx/Y_idx：不对 U_list 施加处理/结果方向约束）
        g, edges, bk = self.generate_combined_graph(V_list, alpha, test_method)

        # 导出因果图
        pyd = GraphUtils.to_pydot(g, labels=V_list)
        final_pyd_string = pyd.to_string()
        self.logger.straight_write(
            "causal_graph",
            f"\n\n====>>> Final Graph <<<=====\n\n{final_pyd_string}\n\n===========\n\n",
        )

        # 记录最终状态
        result["final_V"] = V_list
        # 最终图矩阵与节点顺序（g.graph 的行/列顺序即 V_list 的顺序）
        result["final_graph_matrix"] = np.array(g.graph, copy=True)
        result["node_order"] = list(V_list)
        # 端点编码来源：causallearn PAG 邻接矩阵约定，
        # 与 coatlearn.causality.graph.get_edge_type 等解释函数保持一致
        result["edge_endpoint_encoding"] = (
            "causallearn PAG adjacency convention (see coatlearn.causality.graph.get_edge_type): "
            "entry (i, j) encodes the endpoint of edge i-j on node i's side: "
            "-1 = tail, 1 = arrowhead, 2 = circle, 0 = no edge; "
            "i --> j iff graph[i, j] == -1 and graph[j, i] == 1."
        )
        result["direction_constraints"] = (
            {"required_causal_edge": list(self.config['required_causal_edge']),
             "support": "researcher_prior",
             "required_edge_ci": deepcopy(self._required_edge_ci)}
            if self.config.get('required_causal_edge') is not None else
            "none (no T/Y direction enforced on U_list; pass T_idx/Y_idx to "
            "generate_combined_graph explicitly to add background knowledge)"
        )
        result["annotation_errors"] = deepcopy(self.annotation_errors)
        result["proposal_errors"] = deepcopy(self.proposal_errors)
        result["judge_errors"] = deepcopy(self.judge_errors)
        result["stages"].append(
            {
                "stage": "Stage N+2: Generate Final Causal Graph",
                "final_V_size": len(V_list),
                "final_V_list": V_list,
            }
        )

        return result
