"""Small named-variable interfaces over existing causal-learn and COAT methods."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CausalGraph:
    """A copied endpoint matrix whose rows and columns follow node_order."""

    node_order: tuple[str, ...]
    endpoint_matrix: np.ndarray

    def __post_init__(self):
        names = tuple(self.node_order)
        raw = np.asarray(self.endpoint_matrix)
        if (not names or len(set(names)) != len(names)
                or any(not isinstance(name, str) or not name for name in names)
                or raw.shape != (len(names), len(names))
                or not np.isin(raw, [-1, 0, 1, 2]).all()
                or np.diag(raw).any()
                or not np.array_equal(raw != 0, raw.T != 0)):
            raise ValueError("Graph requires unique node names and a matching PAG endpoint matrix")
        matrix = np.array(raw, dtype=int, copy=True)
        matrix.flags.writeable = False
        object.__setattr__(self, "node_order", names)
        object.__setattr__(self, "endpoint_matrix", matrix)

    @classmethod
    def from_result(cls, result):
        """Read a graph from a run_pag/run_adj result without rerunning discovery."""
        graph = result["graph"]
        if not graph or not graph.get("available"):
            raise ValueError("The result does not contain an available graph")
        return cls(tuple(graph["node_order"]), graph["endpoint_matrix"])

    def to_dict(self):
        return {"node_order": list(self.node_order), "endpoint_matrix": self.endpoint_matrix.tolist()}


def learn_graph(data: pd.DataFrame, *, columns=None, alpha=0.01, method="fisherz", depth=2) -> CausalGraph:
    """Run causal-learn FCI on already measured columns, with no model calls."""
    from causallearn.search.ConstraintBased.FCI import fci

    names = list(data.columns if columns is None else columns)
    values = data.loc[:, names].to_numpy(dtype=float, copy=True)
    if method not in {"mv_fisherz", "chisq", "gsq"}:
        values = np.nan_to_num(values, nan=0.0)
    graph, _ = fci(values, alpha=alpha, depth=depth, independence_test_method=method, verbose=False)
    return CausalGraph(tuple(names), graph.graph)


def find_adjustment_sets(graph: CausalGraph, *, treatment: str, outcome: str) -> dict:
    """Return named adjustment sets and their graphical assumptions."""
    from .adjustment import get_adjustment_sets

    report = {}
    names = list(graph.node_order)
    sets = get_adjustment_sets(names.index(treatment), names.index(outcome), names,
                               graph.endpoint_matrix, verbose=False, diagnostics=report)
    return {"adjustment_sets": [[names[i] for i in sorted(nodes)] for nodes in sets],
            "diagnostics": report}


def estimate_effects(data: pd.DataFrame, graph: CausalGraph, *, treatment: str, outcome: str,
                     factor_definitions: Mapping | None = None, **options) -> dict:
    """Check the fixed graph and use the existing complete-case IPW estimator."""
    from .effects import CausalEffectEstimator
    from .graph import graph_wrapper

    names = list(graph.node_order)
    estimator = CausalEffectEstimator(data, treatment, outcome,
                                      factor_definitions=factor_definitions, **options)
    estimates, by_size = estimator.estimate_all_ates(
        names, graph_wrapper(graph=graph.endpoint_matrix.copy()),
        names.index(treatment), names.index(outcome), verbose=False)
    # Use the same JSON conversion as the discovery result, including NaN handling.
    from coatlearn.api import _json_safe
    return _json_safe({"est_ate_list": estimates, "ate_by_size": by_size,
                       "adjustment_status": estimator.last_adjustment_status,
                       "diagnostics": estimator.diagnostics})
