"""Opt-in minimum edge-deletion hypotheses for nonancestral ADJ graphs.

The original graph is never changed. Common adjustment sets are conditional on
all enumerated deletion hypotheses, not GAC certificates for the original PAG.
"""
from collections import deque

import numpy as np

from coatlearn.causality.adjustment import get_adjustment_sets


class CycleSearchLimit(RuntimeError):
    """Complete minimum-deletion enumeration could not be established."""


def _path(adjacency, source, target):
    pending = deque([(source, [source])])
    seen = {source}
    while pending:
        node, path = pending.popleft()
        for child in np.flatnonzero(adjacency[node]):
            child = int(child)
            if child == target:
                return path + [child]
            if child not in seen:
                seen.add(child)
                pending.append((child, path + [child]))
    return None


def _conflict(matrix):
    """Return edges in one directed or almost-directed cycle witness."""
    directed = (matrix == -1) & (matrix.T == 1)
    for source, target in zip(*np.where(directed)):
        path = _path(directed, int(target), int(source))
        if path:
            return [(int(source), int(target))] + list(zip(path, path[1:]))
    for source, target in zip(*np.where(matrix == 1)):
        # An arrowhead at source forbids source being an ancestor of target.
        path = _path(directed, int(source), int(target))
        if path:
            return [(int(source), int(target))] + list(zip(path, path[1:]))
    return None


def _minimum_deletion_variants(original, protected, search_limit):
    """Search deletion states by size, stopping only after a complete valid layer."""
    frontier = {frozenset()}
    visited = set(frontier)
    variants = []
    minimum = None
    evaluated = 0
    while frontier:
        following = set()
        for removed in sorted(frontier, key=lambda s: tuple(sorted(s))):
            evaluated += 1
            if evaluated > search_limit:
                raise CycleSearchLimit('Minimum-deletion enumeration exceeded search limit')
            candidate = original.copy()
            for a, b in removed:
                candidate[a, b] = candidate[b, a] = 0
            witness = _conflict(candidate)
            if witness is None:
                minimum = len(removed)
                variants.append((removed, candidate))
                continue
            for edge in witness:
                edge = tuple(sorted(edge))
                if edge in protected:
                    continue
                child = removed | {edge}
                if child not in visited:
                    if len(visited) >= search_limit:
                        raise CycleSearchLimit('Minimum-deletion enumeration exceeded search limit')
                    visited.add(child)
                    following.add(child)
        if variants:
            break
        frontier = following
    if not variants:
        raise CycleSearchLimit('No deletion hypothesis preserves required edges')
    return variants, minimum, evaluated


def cycle_break_adjustment_sets(matrix, names, treatment, outcome, *,
                                protected_edges=(), amenability_exempt=False,
                                search_limit=4096):
    """Enumerate every minimum deletion hypothesis, then intersect its sets.

    Search is independent of measurements and ATE. Exhaustion, or conflicts
    involving only protected edges, abstain rather than choose an incomplete
    subset. Ancestor checks are necessary but do not certify PAG realizability.
    """
    original = np.asarray(matrix, dtype=int)
    protected = {tuple(sorted(edge)) for edge in protected_edges}
    variants, minimum, evaluated = _minimum_deletion_variants(original, protected, search_limit)

    common = None
    reports = []
    for removed, candidate in variants:
        criterion = {}
        sets = get_adjustment_sets(treatment, outcome, names, candidate, verbose=False,
                                   diagnostics=criterion, amenability_exempt=amenability_exempt)
        signatures = {tuple(sorted(nodes)) for nodes in sets}
        common = signatures if common is None else common & signatures
        reports.append({
            'deleted_edges': [[names[a], names[b]] for a, b in sorted(removed)],
            'endpoint_matrix': candidate.tolist(), 'adjustment_criterion': criterion,
            'adjustment_sets': [[names[i] for i in nodes]
                                for nodes in sorted(signatures, key=lambda x: (len(x), x))],
        })
    ordered = sorted(common, key=lambda nodes: (len(nodes), nodes))
    report = {
        'enabled': True, 'applied': minimum > 0, 'policy': 'all_minimum_edge_deletions_common_sets',
        'minimum_deleted_edges': minimum, 'variant_count': len(variants),
        'search_complete': True, 'search_states': evaluated, 'search_limit': search_limit,
        'protected_edges': [[names[a], names[b]] for a, b in sorted(protected)],
        'original_endpoint_matrix': original.tolist(), 'node_order': list(names),
        'variants': reports, 'common_adjustment_sets': [[names[i] for i in nodes] for nodes in ordered],
        'gac_certified': False,
        'caveat': 'Edge deletions are researcher-authorized hypotheses, not discovered absent edges. '
                  'Only common sets are estimated; original graph is unchanged. Passing ancestor '
                  'checks does not certify PAG realizability or the original graph.',
    }
    return [list(nodes) for nodes in ordered], report
