"""Generalized adjustment on a fixed PAG, retaining the original endpoint marks."""
import numpy as np
from itertools import combinations
from copy import deepcopy
from scipy import linalg
from .graph import (find_all_paths_pag, check_possibly_directed, is_visible,
                    possible_descendants, get_definite_status_nodes, check_m_sep,
                    print_paths_with_edge_types, get_possible_graph)

class NonAncestralGraphError(ValueError):
    """The supplied graph violates ancestry; GAC is not applicable to it."""


def get_adjustment_sets(T_idx, Y_idx, W_list, graph, verbose=True, enforce=False,
                        *, diagnostics=None, amenability_exempt=False):
    """Enumerate GAC sets for a supplied PAG with single T/Y and no selection.

    Perkovic et al., Generalized Adjustment Criterion (Definition 3.3).
    Validity is conditional on a valid supplied PAG, not proof that the graph
    is causally correct. The original heuristic remains available separately.
    Optional diagnostics describes the three graphical checks in node names.
    """
    if enforce:
        raise ValueError('Adjustment checking cannot force T->Y; supply an explicit graph')
    graph = np.array(graph, copy=True)
    n = len(W_list)
    if (graph.shape != (n, n) or len(set(W_list)) != n
            or not all(isinstance(name, str) and name for name in W_list)
            or not isinstance(T_idx, (int, np.integer))
            or not isinstance(Y_idx, (int, np.integer))
            or not (0 <= T_idx < n and 0 <= Y_idx < n) or T_idx == Y_idx):
        raise ValueError('Invalid graph shape, node names or treatment/outcome indices')
    if (not np.isin(graph, [-1, 0, 1, 2]).all() or np.diag(graph).any()
            or not np.array_equal(graph != 0, graph.T != 0)):
        raise ValueError('Invalid PAG endpoint matrix')
    supported = {(0, 0), (-1, 1), (1, -1), (1, 1), (2, 1), (1, 2), (2, 2)}
    if any((graph[i, j], graph[j, i]) not in supported
           for i in range(n) for j in range(i + 1, n)):
        raise ValueError('Selection edges or unsupported PAG endpoint pair')
    directed = (graph == -1) & (graph.T == 1)
    reach = directed.copy()
    for k in range(n):
        reach |= reach[:, k, None] & reach[None, k, :]
    if np.diag(reach).any() or (reach & (graph == 1)).any():
        raise NonAncestralGraphError('Directed or almost-directed cycle in PAG')
    report = diagnostics if diagnostics is not None else {}
    report.clear()
    report.update(criterion='generalized_adjustment_criterion',
                  assumptions=['supplied graph is a valid PAG/MAG', 'no selection bias'],
                  status='checking')
    paths = find_all_paths_pag(graph, T_idx, Y_idx)
    causal_paths = [p for p in paths if check_possibly_directed(graph, p)]
    failed = [p for p in causal_paths if not is_visible(graph, T_idx, p[1])]
    report['amenable'] = (not failed) or bool(amenability_exempt)
    report['amenability'] = ('exempted_by_researcher' if amenability_exempt
                             else 'checked')
    report['nonamenable_paths'] = [[W_list[i] for i in p] for p in failed]
    if failed and not amenability_exempt:
        report['status'] = 'not_amenable'
        return []

    # Only nodes on possibly causal paths contribute to the forbidden set;
    # Y's descendants are not automatically forbidden when no such path exists.
    on_paths = {node for path in causal_paths for node in path[1:]}
    forbidden = possible_descendants(graph, on_paths)
    eligible = sorted(set(range(len(W_list))) - {T_idx, Y_idx} - forbidden)
    noncausal = [p for p in paths if not check_possibly_directed(graph, p)
                 and get_definite_status_nodes(p, graph)[0]]
    report['forbidden_nodes'] = [W_list[i] for i in sorted(forbidden)]
    report['definite_noncausal_paths'] = [[W_list[i] for i in p] for p in noncausal]
    result = []
    for size in range(len(eligible) + 1):
        for nodes in combinations(eligible, size):
            candidate = set(nodes)
            if all(check_m_sep(candidate, p, graph) for p in noncausal):
                result.append(candidate)
    report['status'] = 'adjustment_sets_found' if result else 'no_adjustment_set'
    report['adjustment_sets'] = [[W_list[i] for i in sorted(nodes)] for nodes in result]
    if verbose:
        print('GAC adjustment sets (conditional on supplied PAG):', report['adjustment_sets'])
    return result


def get_legacy_adjustment_sets(T_idx, Y_idx, W_list, graph, verbose=True, enforce=False):
    """
    Get adjustment sets for the treatment and outcome variables in a causal graph.

    行为约定（最小整理，见 SOURCE_MAP.md）：
    - enforce 默认改为 False：默认不再凭索引强制 graph[T,Y] 的方向；
      需要强加方向时由调用者显式传 enforce=True（仍然只作用于内部的拷贝）。
    - 输入矩阵在本函数内部拷贝，绝不就地修改调用者的图。
    - 空调整集同样进入枚举并按启发式 m-separation 检验：通过时以 [[]] 出现在
      候选中（搜索结果，不是失败兜底）；该启发式忽略邻接/indefinite 路径，
      其输出只是候选列表，不构成 GAC 认证；没有候选通过时返回 []。

    Parameters:
        T_idx (int): Index of the treatment variable.
        Y_idx (int): Index of the outcome variable.
        W_list (list): List of variable names.
        graph (numpy.ndarray): Adjacency matrix of the causal graph.
        verbose (bool): If True, print detailed information.
        enforce (bool): If True, force the direction of the T-Y edge in the
            (internal copy of the) graph before analysis.
    """
    # 拷贝输入矩阵：后续任何修改（包括 enforce）都只作用于本副本
    graph = np.array(graph, copy=True)
    if enforce:
        graph[T_idx, Y_idx] = -1
        graph[Y_idx, T_idx] = 1
    # 找到节点0(email_type)和节点1(device_preference)之间的所有路径
    paths = find_all_paths_pag(graph, T_idx, Y_idx)
    if verbose:
        print_paths_with_edge_types(graph, paths, W_list)

    nodes_on_the_road = set([Y_idx]) # nodes that lies on a possibly directed path from T to Y
    for each_path in paths:
        if check_possibly_directed(graph, each_path):
            for node in each_path[1:]:
                nodes_on_the_road.add(node)

    A = get_possible_graph(graph)
    A = linalg.expm(A)

    forbiden_sets = deepcopy(nodes_on_the_road) # nodes that are a possible desedent of a node in nodes_on_the_road
    for node in nodes_on_the_road:
        forbiden_sets = forbiden_sets.union(set(np.where(A[:,node] !=0)[0]))

    # upper_set = set(np.arange(len(W_list))[(linalg.expm(graph!=0)[:,[T_idx,Y_idx]] > 0).any(1)])
    upper_set = set(np.arange(len(W_list))[(A[[T_idx,Y_idx], :] > 0).any(0)])
    upper_set = upper_set - set([T_idx, Y_idx]) - forbiden_sets - nodes_on_the_road
    must_include = set()
    must_exclude = set()

    
    for path in paths:
        if len(path) <= 2:
            continue
        is_definite_path, definite_collider, definite_non_collider = get_definite_status_nodes(path, graph)

        if not is_definite_path:
            continue

        if check_possibly_directed(graph, path):
            if len(definite_collider-must_exclude) > 0:
                must_exclude = must_exclude.union(definite_collider)
        

    upper_set = upper_set - must_include - must_exclude
    
    possible_adjustment_sets = []
    # set_len 从 0 开始枚举：空调整集同样进入启发式 m-separation 检验，
    # 而不是在找不到调整集时被用作失败兜底（候选，不构成 GAC 认证）。
    for set_len in range(len(upper_set), -1, -1):
        for each_set in combinations(upper_set, set_len):
            each_set = set(each_set).union(must_include)
            valid = True
            for path in paths:
                if len(path) <= 2:
                    continue
                result_from_check_m_sep = check_m_sep(each_set, path, graph)
                if result_from_check_m_sep is None:
                    continue
                if check_possibly_directed(graph, path):
                    valid = valid and (result_from_check_m_sep == False)
                else:
                    valid = valid and (result_from_check_m_sep == True)

                if not valid:
                    break
            
            if valid:
                possible_adjustment_sets.append(each_set) 


    if verbose:
        print('nodes_on_the_road\n\t', [W_list[_] for _ in nodes_on_the_road], sep='')
        print('forbiden_sets\n\t', [W_list[_] for _ in forbiden_sets], sep='')
        print('possible_adjustment_sets:')
        for each_set in possible_adjustment_sets:
            print('\t', [W_list[_] for _ in each_set])

    return possible_adjustment_sets

