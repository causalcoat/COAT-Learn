"""Explicit required-edge support for causal-learn FCI, without vendor patches.

The registered CI adapter uses 0 only as FCI's internal KEEP-EDGE signal for
an explicitly required pair. It is not a measured p-value or causal evidence.
Every other query is delegated unchanged to the configured statistical test.
"""
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.cit import CIT, CIT_Base, register_ci_test


_METHOD = "coatlearn_required_edge"


class _RequiredEdgeCI(CIT_Base):
    def __init__(self, data, *, base_method, required_pair, prior_report, **kwargs):
        super().__init__(data)
        self.base = CIT(data, method=base_method, **kwargs)
        self.method = self.base.method
        self.required_pair = frozenset(required_pair)
        self.prior_report = prior_report

    def __call__(self, x, y, condition_set=None):
        if frozenset((int(x), int(y))) == self.required_pair:
            self.prior_report["protected_ci_calls"] += 1
            return 0.0  # hard prior, NOT a statistical p-value
        return self.base(x, y, condition_set)


register_ci_test(_METHOD, _RequiredEdgeCI)


def fci_with_required_edge(data, source, target, *, alpha, test_method, depth,
                           fci_func, diagnostics=None, tiers=None):
    """Keep and orient one known edge throughout FAS and Possible-D-Sep.

    ``tiers`` (one tier per column, lower = earlier) adds the temporal prior
    that a later variable is not an ancestor of an earlier one. causal-learn
    expresses this per edge by forbidding and reversing the edge, so the
    resulting orientation is stronger than "not an ancestor"; it is recorded
    as a researcher prior, never as CI evidence.
    """
    n = data.shape[1]
    if source == target or not (0 <= source < n and 0 <= target < n):
        raise ValueError("required edge needs distinct valid column indices")
    nodes = [GraphNode(f"X{i + 1}") for i in range(n)]
    knowledge = BackgroundKnowledge().add_required_by_node(nodes[source], nodes[target])
    if tiers is not None:
        if len(tiers) != n:
            raise ValueError("tiers must have one entry per column")
        for index, tier in enumerate(tiers):
            knowledge.add_node_to_tier(nodes[index], int(tier))
    report = diagnostics if diagnostics is not None else {}
    report.update(source_index=source, target_index=target, base_test_method=test_method,
                  support="researcher_prior", ci_evidence_for_required_edge=False,
                  protected_ci_calls=0, protection_scope="all_FCI_CI_queries_for_required_pair",
                  tier_prior=tiers is not None)
    graph, edges = fci_func(
        data, alpha=alpha, depth=depth, independence_test_method=_METHOD,
        verbose=False, background_knowledge=knowledge, base_method=test_method,
        required_pair=(source, target), prior_report=report)
    if (graph.graph[source, target], graph.graph[target, source]) != (-1, 1):
        raise ValueError("FCI did not preserve the explicit required directed edge")
    report["edge_retained"] = True
    return graph, edges, knowledge
