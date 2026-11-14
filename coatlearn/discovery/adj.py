"""
COAT-ADJ: COAT Adjustment Set Discovery and Causal Effect Estimation

该模块调度从文本发现因子、构建因果图和估计效应的完整流程。
因子发现复用 COATMB 和 COATPAG；固定图上的估计由 causality.effects 负责。

主要功能：
- 数据加载和预处理
- 多阶段 COAT 算法执行
- 因果图生成（FCI/LiNGAM）
- 调整集发现
- 因果效应估计（IPW）
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from typing import Dict, Set, List, Tuple, Any, Optional

from coatlearn.discovery.mb import COATMB, default_annotation_batching
from coatlearn.discovery.pag import COATPAG
from coatlearn.causality.effects import CausalEffectEstimator
from coatlearn.causality.graph import find_all_paths_pag, graph_wrapper
from coatlearn.causality.statistics import vif_or_inf
from coatlearn.utils.logging import Logger


from coatlearn.causality.learning import GraphUtils




class COATADJ:
    """
    COAT Adjustment Set Discovery and Causal Effect Estimation

    整合 COATMB 和 COATPAG 的功能，提供完整的因果发现和效应估计流程。

    典型用法：
    ```python
    # 创建实例
    coat_adj = COATADJ(
        llm_proposer=llm_proposer,
        llm_annotator=llm_annotator,
        T_name="email_type",
        Y_name="visit",
        init_kv=init_kv,
        text_values=text_values,
        exp_config=exp_config
    )

    # 运行完整流程
    result = coat_adj.run()

    # 或者分步运行
    coat_adj.run_stage1_mb_discovery()
    coat_adj.run_stage2_path_extension()
    # ...
    ate_list, ate_by_size = coat_adj.estimate_causal_effects()
    ```
    """

    def __init__(
        self,
        llm_proposer,
        llm_annotator,
        T_name: str,
        Y_name: str,
        init_kv: Dict[str, List],
        text_values: np.ndarray,
        exp_config: Dict[str, Any],
        log_root_path: Optional[str] = None,
    ):
        """
        初始化 COAT-ADJ

        Args:
            llm_proposer: LLM 提议器
            llm_annotator: LLM 标注器
            T_name: 处理变量名
            Y_name: 结果变量名
            init_kv: 初始变量字典
            text_values: 文本数据
            exp_config: 实验配置
            log_root_path: 日志根目录
        """
        self.llm_proposer = llm_proposer
        self.llm_annotator = llm_annotator
        self.T_name = T_name
        self.Y_name = Y_name
        self.init_kv = init_kv
        self.text_values = text_values
        self.exp_config = exp_config
        from coatlearn.annotation.prediction import (
            FactorPredictionFilter, validate_prediction_options, validate_observed_definitions,
        )
        validate_observed_definitions(self.exp_config.get("observed_factor_definitions") or {})
        prediction_options = self.exp_config.get("factor_prediction_gate")
        validate_prediction_options(prediction_options)
        self.factor_prediction_filter = (
            FactorPredictionFilter({T_name: init_kv[T_name], Y_name: init_kv[Y_name]}, prediction_options)
            if prediction_options is not None and prediction_options.get("enabled", True) else None
        )
        self.exp_config.setdefault("annotation_batching", default_annotation_batching(llm_annotator))

        # 创建 meta_df
        self.meta_df = pd.DataFrame(init_kv)
        self.meta_df["X"] = text_values

        # 算法状态
        self.W_list = [T_name, Y_name]
        self.S_mark = set()  # 初始化为空，不预置 T 和 Y，允许 Stage 1 发现它们的 MB
        self.round_idx = 0
        self.annotation = deepcopy(init_kv)
        self.factor_pools = {}
        self.proposed_factors_history = []

        # 错误登记（由各 COATMB 子实例汇总而来，供 facade 检查）。
        # 隐私约定：错误只携带异常类型，不带异常消息/模型名/prompt。
        self.annotation_errors = {}  # {(factor_name, sample_idx): exception_type}
        self.proposal_errors = {}
        self.judge_errors = {}

        # Default estimates require GAC on the supplied graph. The explicit
        # legacy option retains the old unverified comparison, not API authorization.
        self.legacy_adjustment_estimates: bool = bool(
            self.exp_config.get("legacy_adjustment_estimates", False)
        )
        self.last_adjustment_status: Optional[str] = None
        self.diagnostics: Dict[str, Any] = {}
        self.direction_constraints: Dict[str, Any] = {}

        # 图对象
        self.g = None
        self.edges = None
        self.bk = None

        # 设置日志：显式可选日志路径。只创建目录并保留文件型 Logger
        # （straight_write）。库运行不劫持 sys.stdout/stderr，也不重配 root
        # logger；需要控制台/文件日志时由调用者自行配置 logging，或显式调用
        # self.logger.attach_root_file_logging(...)（opt-in，见 utils.logging）。
        if log_root_path is None:
            log_root_path = f"./exp_log_folders/log_{exp_config['exp_uid']}"
        self.log_root_path = log_root_path

        if not os.path.exists(log_root_path):
            os.makedirs(log_root_path)
        self.logger = Logger(log_root_path)

        # Legacy F~T/Y R-squared vetoes stay disabled. The explicit prediction
        # policy runs after measurement, before CI, including inside PAG.
        self.TY_quality_gate_list = []
        self.exp_config['ty_quality_gate_policy'] = {
            'extra_ty_gates_enabled': self.factor_prediction_filter is not None,
            'policy': self.factor_prediction_filter.policy() if self.factor_prediction_filter else None,
            'reason': 'explicit single-factor out-of-fold accuracy policy; no R-squared veto',
        }

    def _create_coat_mb_for_target(self, target_name: str) -> COATMB:
        """为单个目标变量创建 COATMB 实例"""
        target_vars = [self.T_name, self.Y_name] + [
            f for f in self.annotation.keys() if f not in [self.T_name, self.Y_name]
        ]

        target_meta_df = pd.DataFrame({k: self.annotation[k] for k in target_vars})
        if "X" not in target_meta_df.columns:
            target_meta_df["X"] = self.text_values

        coat_mb = COATMB(
            llm_proposer=self.llm_proposer,
            llm_annotator=self.llm_annotator,
            target_name=target_name,
            meta_df=target_meta_df,
            text_col="X",
            factor_quality_gate=self.TY_quality_gate_list,
        )

        coat_mb.annotation = deepcopy(self.annotation)
        coat_mb.factor_pools = deepcopy(self.factor_pools)
        coat_mb.proposed_factors_history = deepcopy(self.proposed_factors_history)
        coat_mb.iteration = self.round_idx
        coat_mb.factor_prediction_filter = self.factor_prediction_filter
        coat_mb.set_config(**self.exp_config)

        return coat_mb

    def process_stage(
        self,
        stage_name: str,
        target_nodes: Set[str],
        num_round: int,
        test_method: str,
        alpha: float,
        just_from_pools: bool = False,
    ) -> Tuple[List[str], int]:
        """
        处理单个阶段

        Args:
            stage_name: 阶段名称
            target_nodes: 目标节点集合
            num_round: 每个目标的迭代轮数
            test_method: 独立性测试方法
            alpha: 显著性水平
            just_from_pools: 是否仅从因子池检索

        Returns:
            (W_list, round_idx): 更新后的变量列表和轮次索引
        """
        new_W = []

        for each_target in target_nodes:
            if (not just_from_pools) and (each_target in self.S_mark):
                logging.info(f"Skipping {each_target} as it is already marked.")
                continue

            # 创建 COATMB 实例
            coat_mb = self._create_coat_mb_for_target(each_target)
            coat_mb.V = set([each_target])

            # 执行多轮迭代
            # RuntimeError（LLM API 彻底失败）按既定韧性策略跳过本轮并登记；
            # 其他异常不吞掉、直接抛给 facade，避免阶段失败被冒充为成功。
            for _ in range(num_round):
                previous = set(coat_mb.V)
                self.round_idx += 1
                coat_mb.iteration = self.round_idx - 1
                try:
                    coat_mb.single_round(
                        output_dir=Path(self.log_root_path),
                        alpha=alpha,
                        test_method=test_method,
                        just_from_pools=just_from_pools,
                    )
                except RuntimeError as exc:
                    logging.error('[%s] target=%s round=%s failed: %s',
                                  stage_name, each_target, self.round_idx, type(exc).__name__)
                    self.proposal_errors[
                        f'stage_{stage_name}/target_{each_target}/round_{self.round_idx}'
                    ] = type(exc).__name__
                    break
                if set(coat_mb.V) == previous:
                    break

            new_W = list(set(new_W + list(coat_mb.V)))
            self.S_mark.add(each_target)

            # 合并 annotation
            for factor, values in coat_mb.annotation.items():
                if factor not in self.annotation:
                    self.annotation[factor] = values

            # 更新 factor_pools
            for factor, description in coat_mb.factor_pools.items():
                if factor not in self.factor_pools:
                    self.factor_pools[factor] = description

            # The child began with the complete shared history; replace it
            # with that history plus this target's new rounds, never append it twice.
            self.proposed_factors_history = deepcopy(coat_mb.proposed_factors_history)

            # 汇总子实例的错误登记，供 facade 检查
            self.annotation_errors.update(coat_mb.annotation_errors)
            self.proposal_errors.update(coat_mb.proposal_errors)
            self.judge_errors.update(coat_mb.judge_errors)

        # 检查新因子
        new_W = [_ for _ in new_W if _ not in self.W_list]
        self.W_list = list(dict.fromkeys(
            [self.T_name, self.Y_name] + self.W_list + sorted(new_W)
        ))

        # VIF 清理
        if len(new_W) > 0:
            data_interface = pd.DataFrame(self.annotation)
            self.W_list = GraphUtils.clean_nodes_vif(
                data_interface,
                self.W_list,
                self.T_name,
                self.Y_name,
                self.exp_config.get("vif_threshold", 5.0),
            )

        # 支持度门控：只在 T 的两臂都可观测的因子才能进入图与调整集
        self.W_list = self._apply_support_gate(self.W_list)

        logging.info(f"{stage_name} W_list: {self.W_list}")

        return self.W_list, self.round_idx

    def _apply_support_gate(self, w_list: List[str]) -> List[str]:
        """Drop factors that are unobservable in either treatment arm.

        Only the general template (or any backend that keeps NaN for
        "not mentioned") can be gated this way; the legacy template fills
        missing cells with 0 before this point, so every factor always looks
        observable there. The gate needs a binary 0/1 treatment: without two
        arms there is nothing to compare, so it records ``NOT_APPLICABLE`` and
        keeps the list unchanged.
        """
        if getattr(self, "factor_prediction_filter", None) is not None:
            w_list = self.factor_prediction_filter.filter(w_list, self.annotation, self.factor_pools)
            self.diagnostics['factor_prediction_gate'] = self.factor_prediction_filter.report()
        options = self.exp_config.get("support_gate") or {}
        # On by default: a factor unobservable in one treatment arm cannot
        # support that arm's estimate. Legacy zero-fill makes every factor look
        # observable, so the legacy path is unaffected unless NaNs reach here.
        if not options.get("enabled", True):
            return w_list
        min_rate = float(options.get("min_mention_rate_per_arm", 0.05))
        # An absolute count floor is a per-study choice: a small pilot cannot
        # meet 50 per arm even when every row mentions the factor.
        min_count = int(options.get("min_count_per_arm", 0))
        treatment = np.asarray(self.init_kv[self.T_name], dtype=float)
        arms = np.unique(treatment[np.isfinite(treatment)])
        if len(arms) < 2 or not set(arms.tolist()) <= {0.0, 1.0}:
            # The gate is defined per treatment arm. Graph-only stages accept a
            # non-binary target, and there the gate must not drop everything.
            self.diagnostics.setdefault("support_gate", []).append(
                {"decision": "NOT_APPLICABLE",
                 "reason": "treatment is not a binary 0/1 variable"})
            logging.info("  support gate skipped: treatment is not binary 0/1")
            return w_list
        kept, dropped = [], []
        records = self.diagnostics.setdefault("support_gate", [])
        for name in w_list:
            if name in (self.T_name, self.Y_name):
                kept.append(name)
                continue
            values = self.annotation.get(name)
            if values is None:
                kept.append(name)
                continue
            observed = ~np.isnan(np.asarray(values, dtype=float))
            stats, ok = {}, True
            for arm in (0, 1):
                mask = treatment == arm
                rate = float(observed[mask].mean()) if mask.any() else 0.0
                count = int(observed[mask].sum())
                stats[str(arm)] = {"rate": rate, "count": count}
                ok = ok and rate >= min_rate and count >= min_count
            records.append({"factor": name, "decision": "RETAIN" if ok else "SKIP",
                            **stats})
            if ok:
                kept.append(name)
            else:
                dropped.append(name)
                logging.warning(
                    f"  ⊘ support gate 剔除 {name}: "
                    f"T=1 {stats['1']['rate']:.1%} ({stats['1']['count']} 例), "
                    f"T=0 {stats['0']['rate']:.1%} ({stats['0']['count']} 例)")
        if dropped:
            logging.info(f"support gate 剔除 {len(dropped)} 个因子: {dropped}")
        return kept

    def run_stage1_mb_discovery(self):
        """
        Stage 1: Markov Blanket Discovery

        发现 T 和 Y 的 Markov Blanket
        """
        logging.info("\n" + "=" * 80)
        logging.info("Stage 1: Markov Blanket Discovery")
        logging.info("=" * 80)

        num_round = self.exp_config["num_round"]
        test_method = self.exp_config["test_method"]
        alpha = self.exp_config["alpha"]

        # 发现 T_MB
        self.W_list, self.round_idx = self.process_stage(
            stage_name="Stage 1a: T_MB Discovery",
            target_nodes=set([self.T_name]),
            num_round=num_round,
            test_method=test_method,
            alpha=alpha,
        )
        T_MB = [w for w in self.W_list if w not in [self.T_name, self.Y_name]]

        # 发现 Y_MB
        self.W_list, self.round_idx = self.process_stage(
            stage_name="Stage 1b: Y_MB Discovery",
            target_nodes=set([self.Y_name]),
            num_round=num_round,
            test_method=test_method,
            alpha=alpha,
        )

        logging.info(f"Stage 1 completed. W_list: {self.W_list}")

        # 生成初始因果图
        self._update_graph()

        return self.W_list

    def run_stage2_path_extension(self):
        """Algorithm 3: exhaust path nodes and their possible descendants.

        Recompute the graph and pending set after EACH target's MB search.
        The paper's S_path uses paths between T/Y, not only causal paths.
        """
        from coatlearn.causality.graph import possible_descendants

        self.path_extension_history = []
        while True:
            self._update_graph()
            t_idx, y_idx = self.W_list.index(self.T_name), self.W_list.index(self.Y_name)
            paths = find_all_paths_pag(self.g.graph, t_idx, y_idx)
            on_paths = {node for path in paths for node in path}
            reachable = possible_descendants(self.g.graph, on_paths)
            pending = [name for i, name in enumerate(self.W_list)
                       if i in reachable and name not in self.S_mark]
            if not pending:
                break
            target = pending[0]
            record = {'pending': list(pending), 'target': target,
                      'node_order': list(self.W_list),
                      'endpoint_matrix': self.g.graph.tolist(),
                      'marked_before': sorted(self.S_mark)}
            self.process_stage('Stage 2: Path/descendant extension', {target},
                               self.exp_config['num_round'], self.exp_config['test_method'],
                               self.exp_config['alpha'])
            record['variables_after'] = list(self.W_list)
            self.path_extension_history.append(record)
        return self.W_list

    def run_stage3_uncertain_extension(self, num_round: Optional[int] = None):
        """Algorithm 3 step 4: run existing PAG on the augmented variables.

        Keep the historical method name for callers. Measurements, original
        definitions and proposal history are reused, not regenerated or aliased.
        PAG retains its documented finite extension budget.
        """
        if num_round is None:
            num_round = self.exp_config['num_round']
        structured = list(self.W_list)
        pag = COATPAG(self.llm_proposer, self.llm_annotator, structured,
                      {name: self.annotation[name] for name in structured},
                      self.text_values, str(Path(self.log_root_path) / 'pag_refinement'))
        pag.annotation = deepcopy(self.annotation)
        pag.factor_pools = deepcopy(self.factor_pools)
        pag.proposed_factors_history = deepcopy(self.proposed_factors_history)
        pag.round_idx = self.round_idx
        pag.factor_prediction_filter = self.factor_prediction_filter
        refinement_config = dict(self.exp_config)
        refinement_config.setdefault('fci_depth', 1)  # ADJ's default, not PAG's
        refinement_config['required_causal_edge'] = (
            [self.T_name, self.Y_name]
            if self.exp_config.get('assume_treatment_causes_outcome', False) else None)
        pag.set_config(**refinement_config)
        self.pag_refinement_result = pag.run_algorithm(num_round=num_round)
        self.annotation = deepcopy(pag.annotation)
        self.factor_pools = deepcopy(pag.factor_pools)
        self.proposed_factors_history = deepcopy(pag.proposed_factors_history)
        self.round_idx = pag.round_idx
        self.W_list = list(self.pag_refinement_result['final_V'])
        # PAG refinement rebuilds W_list from its own final_V, so the support
        # gate has to run again or arm-specific factors re-enter the graph.
        self.W_list = self._apply_support_gate(self.W_list)
        self.S_mark.update(structured)
        self.annotation_errors.update(pag.annotation_errors)
        self.proposal_errors.update(pag.proposal_errors)
        self.judge_errors.update(pag.judge_errors)
        for key in ('factor_judge_policy', 'factor_dedup_policy'):
            if key in pag.config:
                self.exp_config[key] = deepcopy(pag.config[key])
        return self.W_list

    def run_stage4_pool_retrieval(self):
        """
        Stage 4: Pool Retrieval

        从因子池检索更多因子
        """
        logging.info("\n" + "=" * 80)
        logging.info("Stage 4: Pool Retrieval")
        logging.info("=" * 80)

        num_round = self.exp_config["num_round"]
        test_method = self.exp_config["test_method"]
        alpha = self.exp_config["alpha"]

        self.W_list, self.round_idx = self.process_stage(
            stage_name="Stage 4: Including more nodes from pools",
            target_nodes=self.W_list,
            num_round=num_round,
            test_method=test_method,
            alpha=alpha,
            just_from_pools=True,
        )

        return self.W_list

    def _update_graph(self):
        """更新因果图

        缺失值保持 NaN：chisq/gsq/mv_fisherz 原生支持缺失（逐对删除），
        其余检验在 generate_combined_graph 内部才按原逻辑填 0。
        这里统一 fillna(0) 会把 general 模板的“未提及”当成真实 0 值。
        """
        self.W_list = self._apply_support_gate(self.W_list)
        data_interface = pd.DataFrame(self.annotation)

        # 在 W_list 中查找 T 和 Y 的实际索引
        # 不能假设 W_list[0] 是 T，W_list[1] 是 Y，因为顺序可能会变化
        try:
            T_idx = self.W_list.index(self.T_name)
            Y_idx = self.W_list.index(self.Y_name)
        except ValueError as e:
            logging.error(
                f"无法在 W_list 中找到变量: {e}\n"
                f"W_list: {self.W_list}\n"
                f"T_name: {self.T_name}, Y_name: {self.Y_name}"
            )
            raise

        # 方向先验为显式 opt-in（默认 False：FCI 不加任何 required 方向边）
        assume_ty = bool(
            self.exp_config.get("assume_treatment_causes_outcome", False)
        )
        factor_tier = bool(
            self.exp_config.get("assume_factors_precede_treatment", False)
        )
        prior_report = {}
        result = GraphUtils.generate_combined_graph(
            data_interface=data_interface,
            W_list=self.W_list,
            alpha=self.exp_config["alpha"],
            test_method=self.exp_config["test_method"],
            T_idx=T_idx,
            Y_idx=Y_idx,
            use_lingam=self.exp_config.get("use_lingam", False),
            fci_depth=self.exp_config.get("fci_depth", 1),
            assume_treatment_causes_outcome=assume_ty,
            prior_diagnostics=prior_report,
            assume_factors_precede_treatment=factor_tier,
        )

        # 方向约束诊断：如实记录进入图学习的方向假设（研究者设定，不是发现）
        self.direction_constraints = {
            "use_lingam": bool(self.exp_config.get("use_lingam", False)),
            "assume_treatment_causes_outcome": assume_ty,
            "T_idx": T_idx,
            "Y_idx": Y_idx,
            "required_edge_ci": prior_report or None,
            "assume_factors_precede_treatment": factor_tier,
            "note": (
                "researcher-injected required adjacency and direction T->Y; "
                "protected from CI deletion, not a discovery or statistical evidence"
                if assume_ty
                else "none (FCI ran without direction background knowledge)"
            ),
        }

        if self.exp_config.get("use_lingam", False):
            self.g = result
        else:
            self.g, self.edges, self.bk = result

    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        运行完整的 COAT-ADJ 流程

        Returns:
            结果字典
        """
        result = {
            "exp_uid": self.exp_config["exp_uid"],
            "W_list_history": [],
        }

        # Stage 1/2/3: 异常直接抛出，由 facade 决定如何处理；不再捕获后冒充成功
        # Stage 1: MB Discovery
        self.run_stage1_mb_discovery()
        result["W_list_history"].append(("Stage 1", deepcopy(self.W_list)))

        # Stage 2: Path Extension
        self.run_stage2_path_extension()
        result["W_list_history"].append(("Stage 2", deepcopy(self.W_list)))

        # Stage 3: Uncertain Extension
        self.run_stage3_uncertain_extension()
        result["W_list_history"].append(("Stage 3", deepcopy(self.W_list)))

        # # Stage 4: Pool Retrieval
        # try:
        #     self.run_stage4_pool_retrieval()
        # except Exception as e:
        #     logging.error(f"Stage 4 失败: {e}")
        # result["W_list_history"].append(("Stage 4", deepcopy(self.W_list)))

        # 更新最终图
        self._update_graph()

        # 导出图
        from causallearn.utils.GraphUtils import GraphUtils

        pyd = GraphUtils.to_pydot(self.g, labels=self.W_list)
        final_pyd_string = pyd.to_string()
        self.logger.straight_write(
            "causal_graph",
            f"\n\n====>>> Final Graph <<<=====\n\n{final_pyd_string}\n\n===========\n\n",
        )

        result['final_W_list'] = list(self.W_list)
        result['direction_constraints'] = deepcopy(self.direction_constraints)
        result['path_extension_history'] = deepcopy(self.path_extension_history)
        result['pag_refinement'] = deepcopy(self.pag_refinement_result)
        return result

    def estimate_causal_effects(
        self, verbose: bool = True
    ) -> Tuple[List[float], Dict[int, List[float]]]:
        """Check GAC on the final PAG, then estimate eligible adjustment sets.

        Return successful values and their grouping by set size; diagnostics
        retains graphical evidence and every set's estimate or skip/failure.
        Explicit legacy mode remains a separately labelled heuristic comparison.
        """
        # 确保图已更新
        if self.g is None:
            self._update_graph()

        # 创建估计器
        data_interface = pd.DataFrame(self.annotation)
        estimator = CausalEffectEstimator(
            data_interface=data_interface,
            T_name=self.T_name,
            Y_name=self.Y_name,
            sample_minimum=self.exp_config.get("sample_minimum", 0.8),
            legacy_adjustment_estimates=self.legacy_adjustment_estimates,
            amenability_exempt=bool(self.exp_config.get("amenability_exempt", False)),
            amenability_exempt_reason=self.exp_config.get("amenability_exempt_reason"),
            factor_definitions=self.factor_pools,
            cell_coverage_min=self.exp_config.get("cell_coverage_min"),
            adjustment_selection=self.exp_config.get(
                "adjustment_selection", "max_cardinality"),
            sparse_cell_warning_below=self.exp_config.get(
                "sparse_cell_warning_below", 20),
            break_cycles=self.exp_config.get('break_cycles', False),
            protected_edges=((self.W_list.index(self.T_name), self.W_list.index(self.Y_name)),)
            if self.exp_config.get('assume_treatment_causes_outcome', False) else (),
        )

        # 在 W_list 中查找 T 和 Y 的实际索引（ADJ 明确保留 T/Y 语义，找不到即失败）
        try:
            T_idx = self.W_list.index(self.T_name)
            Y_idx = self.W_list.index(self.Y_name)
        except ValueError as e:
            raise ValueError(
                f"无法在 W_list 中找到变量: {e}\n"
                f"W_list: {self.W_list}\n"
                f"T_name: {self.T_name}, Y_name: {self.Y_name}"
            ) from e

        # 估计 ATE
        est_ate_list, ate_by_size = estimator.estimate_all_ates(
            W_list=self.W_list, g=self.g, T_idx=T_idx, Y_idx=Y_idx, verbose=verbose
        )

        # 透传状态与诊断信息，供 facade 检查
        self.last_adjustment_status = estimator.last_adjustment_status
        support_records = self.diagnostics.get('support_gate')
        self.diagnostics = deepcopy(estimator.diagnostics)
        if self.factor_prediction_filter is not None:
            self.diagnostics['factor_prediction_gate'] = self.factor_prediction_filter.report()
        if support_records:
            # Discovery-side evidence must survive the estimator's own report.
            self.diagnostics['support_gate'] = deepcopy(support_records)

        return est_ate_list, ate_by_size

    def run(self) -> Dict[str, Any]:
        """
        运行完整流程（包括因果效应估计）

        Returns:
            结果字典，包含：
            - W_list_history: 各阶段的 W_list
            - final_W_list: 最终变量列表
            - est_ate_list: ATE 估计列表
            - ate_by_size: 按调整集大小分组的 ATE
        """
        # 运行完整流程
        result = self.run_full_pipeline()

        # 估计因果效应
        est_ate_list, ate_by_size = self.estimate_causal_effects()

        result["est_ate_list"] = est_ate_list
        result["ate_by_size"] = ate_by_size
        result["last_adjustment_status"] = self.last_adjustment_status
        result["legacy_adjustment_estimates"] = self.legacy_adjustment_estimates
        result["diagnostics"] = deepcopy(self.diagnostics)
        result["direction_constraints"] = deepcopy(self.direction_constraints)
        result["annotation_errors"] = deepcopy(self.annotation_errors)
        result["proposal_errors"] = deepcopy(self.proposal_errors)
        result["judge_errors"] = deepcopy(self.judge_errors)

        return result
