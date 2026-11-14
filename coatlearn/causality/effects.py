"""Effect estimation on a fixed graph and measured factors, without discovery.

The estimator owns adjustment-set checks, complete-case IPW, numerical gates
and primary-set selection. Optional cycle hypotheses live in graph_postprocessing;
none of these operations creates a model client or changes the supplied graph.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from coatlearn.annotation.types import UnsupportedMeasurement
from coatlearn.annotation.definitions import factor_definition_to_spec
from coatlearn.causality.adjustment import NonAncestralGraphError, get_adjustment_sets
from coatlearn.causality.graph import is_visible
from coatlearn.causality.statistics import is_ipw_valid


class CausalEffectEstimator:
    """
    因果效应估计器

    功能：
    - 查找调整集
    - 记录 IPW 样本覆盖与加权平衡指标
    - 估计 ATE（IPW 方法）
    """

    def __init__(
        self,
        data_interface: pd.DataFrame,
        T_name: str,
        Y_name: str,
        sample_minimum: float = 0.8,
        legacy_adjustment_estimates: bool = False,
        amenability_exempt: bool = False,
        amenability_exempt_reason: Optional[str] = None,
        factor_definitions: Optional[Dict[str, Any]] = None,
        cell_coverage_min: Optional[float] = None,
        adjustment_selection: str = "max_cardinality",
        sparse_cell_warning_below: int = 20,
        break_cycles: bool = False,
        protected_edges=(),
    ):
        """
        初始化因果效应估计器

        Args:
            data_interface: 数据接口 DataFrame
            T_name: 处理变量名
            Y_name: 结果变量名
            sample_minimum: 最小样本比例阈值
            legacy_adjustment_estimates: 默认False，先检查给定PAG的GAC，
                仅对通过的集合估计；逐集样本与平衡检查不通过时弃算。
                True保留旧启发式对照，结果明确标为legacy/unverified。
                图上的有效性不认证实际因果图、测量或估计假设。
            factor_definitions: 因子名→定义原文，用于解析声明档位以统计格子
                覆盖度；缺省时跳过格子指标，不因缺失定义而弃算。
            cell_coverage_min: 集合级组合覆盖度下限；None 表示不启用。
            adjustment_selection: 主集选择策略。``max_cardinality`` 保持旧口径
                （最大基数），``size_then_propensity_range`` 先按基数、再按倾向
                值区间宽度降序选第一个。
            sparse_cell_warning_below: 选中集合最小格样本数低于此值时只记录
                告警，不弃算。
            break_cycles: 默认关闭。开启后只估计所有最少删边假设共同允许的集。
                原图保留；去环假设不认证原图或其因果效应。
            protected_edges: 去环时不可删除的边，以原图节点索引对表示。
        """
        if adjustment_selection not in (
                "max_cardinality", "size_then_propensity_range"):
            raise ValueError(
                "adjustment_selection must be 'max_cardinality' or "
                f"'size_then_propensity_range', got {adjustment_selection!r}")
        self.data_interface = data_interface
        self.T_name = T_name
        self.Y_name = Y_name
        self.sample_minimum = sample_minimum
        self.legacy_adjustment_estimates = bool(legacy_adjustment_estimates)
        self.amenability_exempt = bool(amenability_exempt)
        self.amenability_exempt_reason = amenability_exempt_reason
        self.factor_definitions = dict(factor_definitions or {})
        self.cell_coverage_min = cell_coverage_min
        self.adjustment_selection = adjustment_selection
        self.sparse_cell_warning_below = int(sparse_cell_warning_below)
        if not isinstance(break_cycles, bool):
            raise ValueError('break_cycles must be a bool')
        if break_cycles and legacy_adjustment_estimates:
            raise ValueError('break_cycles cannot be combined with legacy adjustment estimates')
        self.break_cycles = break_cycles
        self.protected_edges = tuple(protected_edges)
        self._level_cache: Dict[str, Optional[List[float]]] = {}

        # GAC status separates graphical abstention, numerical skips and
        # successful estimates. Legacy statuses are retained only in explicit
        # comparison mode. Per-set evidence lives in diagnostics['sets'].
        self.last_adjustment_status: Optional[str] = None
        self.diagnostics: Dict[str, Any] = {}

    def find_adjustment_sets(
        self, W_list: List[str], g, T_idx: int = 0, Y_idx: int = 1, verbose: bool = True
    ) -> List[List[int]]:
        """Return GAC sets conditional on the supplied PAG, with named evidence.

        Unsupported/invalid graph input raises ValueError. No candidate is
        substituted for failure; an empty valid set is represented by [[]].
        """
        exempt = bool(self.amenability_exempt)
        if exempt and not self.amenability_exempt_reason:
            raise ValueError(
                "amenability_exempt requires amenability_exempt_reason")
        report = {}
        sets = get_adjustment_sets(T_idx, Y_idx, W_list, g.graph, verbose,
                                   diagnostics=report, amenability_exempt=exempt)
        if exempt:
            report['amenability_exempt_reason'] = self.amenability_exempt_reason
            report['gac_certified'] = False
        self.diagnostics['adjustment_criterion'] = report
        self.diagnostics['candidate_adjustment_sets'] = report.get('adjustment_sets', [])
        if not sets:
            self.last_adjustment_status = (
                'gac_not_amenable_abstain' if report['status'] == 'not_amenable'
                else 'gac_no_adjustment_set')
        else:
            self.last_adjustment_status = 'gac_adjustment_sets_found'
        return [sorted(nodes) for nodes in sets]

    def estimate_ate_with_ipw(
        self,
        adjustment_set: List[str],
        verbose: bool = True,
        empty_set_legacy_allowed: bool = False,
    ) -> Optional[float]:
        """
        使用 IPW 估计单个候选调整集（legacy 对照路径）

        本方法不做任何可识别性认证；估计值是否可以输出由调用方
        （estimate_all_ates）的 legacy_adjustment_estimates 开关与诊断决定。

        Args:
            adjustment_set: 候选调整集（变量名列表）
            verbose: 是否打印详细信息
            empty_set_legacy_allowed: 是否允许对空候选集计算未调整均值差。
                仅在 legacy 对照模式且该空集来自启发式候选列表时由
                estimate_all_ates 传 True；默认 False，直接调用传空集时弃算
                （未调整均值差不作为因果效应输出）。

        Returns:
            ATE 数值（失败或弃算时返回 None；逐集技术状态见
            self.diagnostics["sets"]，不是科学有效性断言）
        """
        set_diagnostics: Dict[str, Any] = {"adjustment_set": list(adjustment_set)}
        self.diagnostics.setdefault("sets", []).append(set_diagnostics)

        if len(adjustment_set) == 0:
            if not empty_set_legacy_allowed:
                set_diagnostics["status"] = "empty_set_not_allowed_abstain"
                if verbose:
                    logging.warning(
                        "Empty adjustment set not allowed in this mode; refusing "
                        "to report the unadjusted mean difference as a causal "
                        "effect."
                    )
                return None

            if verbose:
                logging.info(
                    "Legacy mode: computing unadjusted mean difference for the "
                    "empty candidate set (unverified)"
                )

            T_values = self.data_interface[self.T_name].values
            Y_values = self.data_interface[self.Y_name].values

            ate = np.mean(Y_values[T_values == 1]) - np.mean(Y_values[T_values == 0])
            set_diagnostics.update(
                {"status": "legacy_naive_difference_unverified", "ate": ate}
            )
            return ate

        # 检查缺失值
        valid_indicator = ~self.data_interface[adjustment_set].isna().values
        mask = valid_indicator.all(1).reshape(-1)

        if min(mask.mean(), valid_indicator.mean(0).min()) < self.sample_minimum:
            set_diagnostics["status"] = "skipped_insufficient_data"
            if verbose:
                logging.warning(
                    f"{adjustment_set} - Too many missing values "
                    f"[{mask.mean()} {valid_indicator.mean(0).min()}], skipping"
                )
            return None

        # 准备数据
        X = self.data_interface[adjustment_set].values[mask, :]
        T_values = self.data_interface[self.T_name].values[mask]
        Y_values = self.data_interface[self.Y_name].values[mask]

        try:
            # IPW 平衡性诊断检查
            ipw_valid, ipw_info = is_ipw_valid(X, T_values, Y_values)
            if not ipw_valid:
                set_diagnostics["status"] = "skipped_ipw_balance_check_failed"
                set_diagnostics["ipw_info"] = ipw_info
                if verbose:
                    logging.warning(
                        f"{adjustment_set} - IPW check failed "
                        f"({ipw_info}), skipping"
                    )
                return None

            # 拟合倾向性评分模型
            model = LogisticRegression(random_state=0).fit(X, T_values)
            e = model.predict_proba(X)[:, 1]

            # 裁剪倾向性评分
            e = np.clip(e, 0.01, 0.99)

            # 计算 ATE
            ate = (Y_values * T_values / e - Y_values * (1 - T_values) / (1 - e)).mean()

            set_diagnostics.update(
                {
                    "status": "estimated_legacy_unverified",
                    "ate": ate,
                    "n": int(sum(mask)),
                }
            )

            if verbose:
                logging.info(
                    f"Adjustment set {adjustment_set} (legacy, unverified): "
                    f"ATE = {ate:.4f}, sample = {sum(mask)}"
                )

            return ate

        except Exception as exc:
            # 隐私约定：诊断中的错误只携带异常类型
            set_diagnostics["status"] = "estimation_failed"
            set_diagnostics["error"] = type(exc).__name__
            if verbose:
                logging.error(f"Error in model fitting for {adjustment_set}: {exc}")
            return None

    def _declared_levels(self, name: str) -> Optional[List[float]]:
        """Declared value grid of one factor, parsed from its saved definition.

        Score factors are annotated as probability-weighted expectations, so the
        observed values are continuous even though the rubric declares a finite
        grid; callers snap each value to its nearest declared level. Returns
        ``None`` when no definition exists or it cannot be parsed, and callers
        then skip the cell diagnostics rather than skipping the estimate.
        """
        if name in self._level_cache:
            return self._level_cache[name]
        levels: Optional[List[float]] = None
        definition = self.factor_definitions.get(name)
        if definition is not None:
            try:
                spec = factor_definition_to_spec(name, definition)
                raw = (spec.choice_values.values()
                       if getattr(spec, "choice_values", None)
                       else getattr(spec, "value_map", ()) or ())
                parsed = sorted({float(v) for v in raw})
                levels = parsed or None
            except (UnsupportedMeasurement, TypeError, ValueError):
                levels = None
        self._level_cache[name] = levels
        return levels

    def _cell_diagnostics(self, names: List[str], valid: np.ndarray) -> Dict[str, Any]:
        """Count how many declared level combinations the usable rows cover.

        ``cell_coverage`` is observed combinations over the product of declared
        level counts; ``min_cell_n`` is the smallest observed combination. Both
        describe positivity (does every declared cell have data), not balance.
        """
        if not names:
            return {'cell_coverage': 1.0, 'observed_cells': 1,
                    'declared_cells': 1, 'min_cell_n': int(valid.sum()),
                    'mean_cell_n': float(valid.sum())}
        grids, declared_total = [], 1
        for name in names:
            declared = self._declared_levels(name)
            if not declared:
                return {'cell_coverage': None, 'observed_cells': None,
                        'declared_cells': None, 'min_cell_n': None,
                        'mean_cell_n': None,
                        'cell_note': f'no declared levels for {name}'}
            grid = np.asarray(declared, dtype=float)
            values = self.data_interface[name].to_numpy(dtype=float)
            snapped = grid[np.abs(values[:, None] - grid[None, :]).argmin(axis=1)]
            grids.append(snapped)
            declared_total *= len(grid)
        matrix = np.column_stack(grids)
        mask = np.isfinite(matrix).all(axis=1) & valid
        if not mask.any():
            return {'cell_coverage': 0.0, 'observed_cells': 0,
                    'declared_cells': declared_total, 'min_cell_n': 0,
                    'mean_cell_n': 0.0}
        cells, counts = np.unique(matrix[mask], axis=0, return_counts=True)
        observed = int(len(cells))
        return {'cell_coverage': observed / declared_total,
                'observed_cells': observed,
                'declared_cells': declared_total,
                'min_cell_n': int(counts.min()),
                'mean_cell_n': float(counts.mean())}

    def _select_primary(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pick the reported adjustment set among graph-valid estimated sets.

        ``max_cardinality`` keeps the historical choice. The marketing-config
        policy orders by cardinality first and by propensity spread second: a
        confounder is the admissible variable that most separates the arms, so
        a wider spread is the tie-breaker among equally large sets.
        """
        estimated = [r for r in records if r.get('status') == 'estimated']
        if not estimated:
            return {'policy': self.adjustment_selection,
                    'status': 'no_estimated_set', 'adjustment_set': None,
                    'estimate': None}
        if self.adjustment_selection == "max_cardinality":
            ordered = sorted(estimated, key=lambda r: -len(r['adjustment_set']))
        else:
            ordered = sorted(
                estimated,
                key=lambda r: (-len(r['adjustment_set']),
                               -(r.get('propensity_range') or 0.0)))
        best = ordered[0]
        min_cell = best.get('min_cell_n')
        return {
            'policy': self.adjustment_selection,
            'status': 'selected',
            'adjustment_set': list(best['adjustment_set']),
            'estimate': best['estimate'],
            'size': len(best['adjustment_set']),
            'n': best['n'],
            'coverage': best['coverage'],
            'cell_coverage': best.get('cell_coverage'),
            'min_cell_n': min_cell,
            'propensity_range': best.get('propensity_range'),
            'max_weight': best.get('max_weight'),
            'ess_ratio': best.get('ess_ratio'),
            'max_smd': best.get('max_smd'),
            'sparse_cell_warning': (min_cell is not None
                                    and min_cell < self.sparse_cell_warning_below),
            'caveat': ('Selection ranks graph-valid estimated sets; it does '
                       'not certify identification or the discovered graph.'),
        }

    def _estimate_gac_set(self, names: List[str]) -> Dict[str, Any]:
        """Estimate one already-checked set; diagnostics use the actual weights."""
        row = {'adjustment_set': list(names), 'estimate': None,
               'kind': 'ipw_gac' if names else 'unadjusted_mean_difference_gac'}
        values = self.data_interface[[self.T_name, self.Y_name] + names].to_numpy(dtype=float)
        valid = np.isfinite(values).all(axis=1)
        row.update(n=int(valid.sum()), coverage=float(valid.mean()) if len(valid) else 0.0)
        if not valid.any() or row['coverage'] < self.sample_minimum:
            row['status'] = 'skipped_insufficient_data'
            return row
        row.update(self._cell_diagnostics(names, valid))
        if (self.cell_coverage_min is not None
                and row.get('cell_coverage') is not None
                and row['cell_coverage'] < self.cell_coverage_min):
            row['status'] = 'skipped_cell_coverage'
            return row
        values = values[valid]
        treatment, outcome = values[:, 0], values[:, 1]
        row['treatment_counts'] = {str(v): int((treatment == v).sum()) for v in (0, 1)}
        if not np.isin(treatment, [0, 1]).all() or not all(row['treatment_counts'].values()):
            row['status'] = 'skipped_treatment_support'
            return row
        if not names:
            row.update(status='estimated', estimate=float(
                outcome[treatment == 1].mean() - outcome[treatment == 0].mean()))
            return row

        covariates = values[:, 2:]
        try:
            # Same logistic model and clipping as the legacy estimate; unlike
            # legacy, the balance diagnostic evaluates THESE estimation weights.
            model = LogisticRegression(random_state=0).fit(covariates, treatment)
            propensity = model.predict_proba(covariates)[:, 1]
            if not np.isfinite(propensity).all():
                row['status'] = 'skipped_nonfinite_propensity'
                return row
            clipped = np.clip(propensity, 0.01, 0.99)
            weights = treatment / clipped + (1 - treatment) / (1 - clipped)
            treated, control = treatment == 1, treatment == 0
            difference = np.abs(np.average(covariates[treated], axis=0, weights=weights[treated])
                                - np.average(covariates[control], axis=0, weights=weights[control]))
            scale = np.sqrt((covariates[treated].var(axis=0) + covariates[control].var(axis=0)) / 2)
            smd = np.divide(difference, scale, out=np.zeros_like(difference), where=scale > 0)
            smd[(scale == 0) & (difference > 0)] = np.inf
            row.update(propensity_model='LogisticRegression(random_state=0)',
                       propensity_min=float(propensity.min()), propensity_max=float(propensity.max()),
                       propensity_clip=[0.01, 0.99], clipped_count=int((clipped != propensity).sum()),
                       balance_threshold=0.15,
                       max_smd=float(smd.max()) if np.isfinite(smd).all() else None)
            # 1/e weighting concentrates information: record how much of n the
            # weighted sample still represents, next to the propensity spread.
            ess = float(weights.sum() ** 2 / (weights ** 2).sum())
            row.update(propensity_range=float(propensity.max() - propensity.min()),
                       max_weight=float(weights.max()), ess=ess,
                       ess_ratio=ess / len(treatment))
            if not np.isfinite(smd).all() or smd.max() > 0.15:
                row['status'] = 'skipped_ipw_balance_check_failed'
                return row
            estimate = float((outcome * treatment / clipped
                              - outcome * (1 - treatment) / (1 - clipped)).mean())
            if not np.isfinite(estimate):
                row['status'] = 'skipped_nonfinite_estimate'
                return row
            row.update(status='estimated', estimate=estimate)
        except Exception as exc:
            row.update(status='estimation_failed', error_type=type(exc).__name__)
        return row

    def _estimate_checked_sets(self, W_list, g, T_idx, Y_idx, verbose):
        """Apply graph checks and the same numerical policy to every eligible set."""
        self.diagnostics = {
            'W_list': list(W_list), 'T_idx': T_idx, 'Y_idx': Y_idx,
            'legacy_adjustment_estimates': False,
            'caveat': 'Graphical validity is conditional on the supplied PAG and its assumptions.',
        }
        try:
            sets = self.find_adjustment_sets(W_list, g, T_idx, Y_idx, verbose)
        except NonAncestralGraphError:
            # Finite-sample discovery can violate ancestral assumptions.
            # Preserve the graph and decline GAC/estimation; do not hide
            # unrelated malformed-input or implementation exceptions.
            self.last_adjustment_status = 'gac_invalid_graph_abstain'
            self.diagnostics.update(
                adjustment_criterion={
                    'criterion': 'generalized_adjustment_criterion',
                    'assumptions': ['supplied graph is a valid PAG/MAG', 'no selection bias'],
                    'status': 'invalid_graph',
                    'reason': 'directed_or_almost_directed_cycle',
                },
                candidate_adjustment_sets=None,
                sets=None,
            )
            if not self.break_cycles:
                return [], {}
            from coatlearn.causality.postprocessing import cycle_break_adjustment_sets, CycleSearchLimit
            try:
                sets, processing = cycle_break_adjustment_sets(
                    g.graph, W_list, T_idx, Y_idx,
                    protected_edges=self.protected_edges,
                    amenability_exempt=self.amenability_exempt)
            except CycleSearchLimit as exc:
                self.diagnostics['graph_postprocessing'] = {
                    'enabled': True, 'applied': False, 'search_complete': False,
                    'status': 'abstained', 'reason': str(exc), 'gac_certified': False}
                return [], {}
            self.diagnostics['graph_postprocessing'] = processing
            self.diagnostics['adjustment_criterion'] = {
                'criterion': 'generalized_adjustment_criterion',
                'status': 'adjustment_sets_found' if sets else 'no_adjustment_set',
                'scope': 'intersection across all minimum edge-deletion hypotheses',
                'original_graph_status': 'invalid_graph',
                'amenability': 'exempted_by_researcher' if self.amenability_exempt else 'checked_per_variant',
                'amenability_exempt_reason': self.amenability_exempt_reason,
                'gac_certified': False,
                'assumptions': ['each hypothesis is a valid PAG/MAG', 'no selection bias'],
            }
            self.diagnostics['candidate_adjustment_sets'] = processing['common_adjustment_sets']
            self.diagnostics['caveat'] = processing['caveat']
            self.last_adjustment_status = 'cycle_break_no_common_set'
        processed = self.diagnostics.get('graph_postprocessing', {}).get('applied', False)
        self.diagnostics['is_visible'] = (None if processed else bool(is_visible(g.graph, T_idx, Y_idx)))
        self.diagnostics['sets'] = []
        if not sets:
            return [], {}
        records = [self._estimate_gac_set([W_list[i] for i in nodes]) for nodes in sets]
        self.diagnostics['sets'] = records
        self.diagnostics['primary_adjustment_summary'] = self._select_primary(records)
        # Select size from ALL graph-valid sets, before numerical skips.
        # A smaller successful set must not replace a failed largest set.
        largest_size = max(len(nodes) for nodes in sets)
        largest = [row for row in records
                   if len(row['adjustment_set']) == largest_size]
        largest_values = [row['estimate'] for row in largest
                          if row['status'] == 'estimated']
        self.diagnostics['largest_adjustment_summary'] = {
            'policy': 'maximum_cardinality_gac',
            'status': ('estimated' if len(largest_values) == len(largest)
                       else 'partially_estimated' if largest_values
                       else 'all_sets_skipped'),
            'max_size': largest_size,
            'n_sets': len(largest),
            'n_estimated': len(largest_values),
            'mean_ate': float(np.mean(largest_values)) if largest_values else None,
            'adjustment_sets': [row['adjustment_set'] for row in largest],
        }
        successful = [row for row in records if row['status'] == 'estimated']
        by_size = {}
        for row in successful:
            by_size.setdefault(len(row['adjustment_set']), []).append(row['estimate'])
        self.diagnostics['n_estimates'] = len(successful)
        prefix = 'cycle_break_' if processed else 'gac_'
        self.last_adjustment_status = prefix + (
            'estimated' if len(successful) == len(records)
            else 'partially_estimated' if successful else 'all_sets_skipped')
        if processed:
            for row in records:
                row['kind'] = 'ipw_cycle_break_hypotheses' if row['adjustment_set'] else 'unadjusted_cycle_break_hypotheses'
            self.diagnostics['primary_adjustment_summary']['caveat'] = self.diagnostics['caveat']
        return [row['estimate'] for row in successful], by_size

    def estimate_all_ates(
        self,
        W_list: List[str],
        g,
        T_idx: int = 0,
        Y_idx: int = 1,
        verbose: bool = True,
        legacy_adjustment_estimates: Optional[bool] = None,
    ) -> Tuple[List[float], Dict[int, List[float]]]:
        """Estimate only sets passing GAC on the supplied PAG by default.

        Returns successful values and their grouping by set size. Diagnostics
        keeps every set's identity/status, including numerical skips/failures.
        An explicit legacy override retains the old, unverified comparison.
        Neither mode certifies the discovered graph or estimation assumptions.
        """
        if (not isinstance(T_idx, (int, np.integer))
                or not isinstance(Y_idx, (int, np.integer))
                or not (0 <= T_idx < len(W_list) and 0 <= Y_idx < len(W_list))
                or W_list[T_idx] != self.T_name or W_list[Y_idx] != self.Y_name):
            raise ValueError('Graph treatment/outcome indices do not match estimator variables')
        if legacy_adjustment_estimates is None:
            legacy_adjustment_estimates = self.legacy_adjustment_estimates
        legacy_mode = bool(legacy_adjustment_estimates)
        if not legacy_mode:
            return self._estimate_checked_sets(W_list, g, T_idx, Y_idx, verbose)

        self.last_adjustment_status = None
        self.diagnostics = {
            "W_list": list(W_list),
            "T_idx": T_idx,
            "Y_idx": Y_idx,
            "legacy_adjustment_estimates": legacy_mode,
            "caveat": (
                "get_adjustment_sets is a legacy heuristic (ignores adjacent "
                "and indefinite paths); its candidates are NOT a GAC certificate."
            ),
        }

        # 检查因果效应可见性
        # 注意：单条 T->Y 边的 is_visible 不是所有图上的完整可识别性判据。
        # “不可见”只作为弃算信号记录（不是证明不可识别），
        # 状态记录在 last_adjustment_status / diagnostics 供 facade 检查。
        visible = is_visible(g.graph, T_idx, Y_idx)
        self.diagnostics["is_visible"] = bool(visible)
        if not visible:
            self.last_adjustment_status = "not_visible_abstain"
            if verbose:
                logging.warning(
                    f"is_visible({W_list[T_idx]} -> {W_list[Y_idx]}) is False. "
                    "Single-edge visibility is not a complete identifiability "
                    "test, so this is recorded as an abstention (not a proof of "
                    "non-identifiability). Returning empty results."
                )
            return [], {}

        from coatlearn.causality.adjustment import get_legacy_adjustment_sets

        candidate_sets = get_legacy_adjustment_sets(T_idx, Y_idx, W_list, g.graph, verbose)
        self.diagnostics['candidate_adjustment_sets'] = [
            [W_list[i] for i in sorted(nodes)] for nodes in candidate_sets
        ]
        if not candidate_sets:
            self.last_adjustment_status = 'no_adjustment_set_found'
            return [], {}

        # legacy 对照模式：计算但明确标记 legacy/unverified
        est_ate_list = []
        ate_by_size = {}

        for each_candidate_set in candidate_sets:
            selected_node_names = [W_list[_] for _ in each_candidate_set]

            ate = self.estimate_ate_with_ipw(
                selected_node_names,
                verbose,
                empty_set_legacy_allowed=(len(each_candidate_set) == 0),
            )

            if ate is not None:
                est_ate_list.append(ate)

                # 按候选集大小分组
                if len(selected_node_names) not in ate_by_size:
                    ate_by_size[len(selected_node_names)] = []
                ate_by_size[len(selected_node_names)].append(ate)

        if est_ate_list:
            self.last_adjustment_status = "legacy_heuristic_estimates_unverified"
            self.diagnostics["n_estimates"] = len(est_ate_list)
        else:
            self.last_adjustment_status = "legacy_all_sets_failed"
            self.diagnostics["n_estimates"] = 0

        # 打印汇总统计
        if est_ate_list and verbose:
            logging.info(
                f"\nAverage legacy (unverified) ATE estimate: "
                f"{np.mean(est_ate_list):.4f}"
            )
            logging.info(f"Standard deviation: {np.std(est_ate_list):.4f}")
            logging.info(
                f"ATE range: {min(est_ate_list):.4f} to {max(est_ate_list):.4f}"
            )
            logging.info("\nLegacy ATE estimates by size of adjustment set:")
            for size, estimates in ate_by_size.items():
                logging.info(
                    f"Size {size}: {len(estimates)} estimates, "
                    f"mean = {np.mean(estimates):.4f}, "
                    f"std = {np.std(estimates):.4f}"
                )
        elif verbose:
            logging.warning(
                "No legacy ATE estimates were computed (see "
                "last_adjustment_status / diagnostics)."
            )

        return est_ate_list, ate_by_size
