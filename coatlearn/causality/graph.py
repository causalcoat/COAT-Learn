"""PAG endpoint interpretation and traversal in causal-learn's encoding."""
import numpy as np
from typing import List
from scipy import linalg
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

class graph_wrapper:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def get_possible_graph(G):
    """
        Input: PAG from FCI. see https://causal-learn.readthedocs.io/
        Output: a possible graph A.
            A[j,i] = 1: node i is a possible cause of node j, i.e., i --> j is possible
    """
    assert G.shape[0] == G.shape[1]
    n, _ = G.shape
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if (G[i,j]==-1) and (G[j,i]==1): # i --> j
                A[j,i] = 1
            if (G[i,j]==2) and (G[j,i]==1): # i o-> j
                A[j,i] = 1
            if (G[i,j]==2) and (G[j,i]==2): # i o-o j
                A[j,i] = 1
                A[i,j] = 1

    return A


def get_directed_graph(G):
    """
        Input: PAG from FCI. see https://causal-learn.readthedocs.io/
        Output: a directed graph A.
            A[j,i] = 1: node i is a cause of node j, i.e., i --> j is sure
    """
    assert G.shape[0] == G.shape[1]
    n, _ = G.shape
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if (G[i,j]==-1) and (G[j,i]==1): # i --> j
                A[j,i] = 1

    return A


def find_all_paths_pag(graph, idx_1, idx_2, max_path_length=None):
    """
    查找PAG图中两个节点之间的所有路径
    
    参数:
        graph: PAG图的邻接矩阵表示
        idx_1: 起始节点索引
        idx_2: 目标节点索引
        max_path_length: 可选，限制路径的最大长度
    
    返回:
        list: 包含所有路径的列表，每条路径是一个节点索引列表
    """
    n = graph.shape[0]  # 节点数量
    visited = [False] * n
    path = []
    all_paths = []
    
    def dfs(current, end, path, visited):
        path.append(current)
        visited[current] = True
        
        if current == end:
            all_paths.append(path.copy())
        else:
            if max_path_length is not None and len(path) >= max_path_length:
                pass
            else:
                for neighbor in range(n):
                    # 检查是否存在任何类型的边
                    if (graph[current, neighbor] != 0 or graph[neighbor, current] != 0) and not visited[neighbor]:
                        dfs(neighbor, end, path, visited)
        
        path.pop()
        visited[current] = False
    
    dfs(idx_1, idx_2, path, visited)
    return all_paths


def get_edge_type(graph, from_node, to_node):
    """
    精确获取PAG图中边的类型
    
    参数:
        graph: adjacency matrix of the PAG graph
        from_node: 起始节点索引
        to_node: 目标节点索引
    
    返回:
        str: 边类型描述
    """
    f_to_t = graph[from_node, to_node] # G[A,B]
    t_to_f = graph[to_node, from_node]
    
    if f_to_t == -1 and t_to_f == 1:
        return "-->"  # A → B (A causes B)
    elif f_to_t == 1 and t_to_f == -1:
        return "<--"
    elif f_to_t == 2 and t_to_f == 1:
        return "∘->" 
    elif f_to_t == 1 and t_to_f == 2:
        return "<-∘"
    elif f_to_t == 2 and t_to_f == 2:
        return "∘─∘"  # A ∘─∘ B (no d-separation)
    elif f_to_t == 1 and t_to_f == 1:
        return "<─>"  # A ∘─∘ B (latent common cause)
    else:
        return "?"  # 未知类型


def get_definite_status_nodes(path, graph):
    assert len(path) >= 2
    if len(path) == 2:
        return True, set(), set()

    is_definite_path = True
    definite_collider = set()   
    definite_non_collider = set()
    for each_node_id in path[1:-1]:
        left_node_id = path[path.index(each_node_id) - 1]
        right_node_id = path[path.index(each_node_id) + 1]
        left_edge_type = get_edge_type(graph, left_node_id, each_node_id)
        right_edge_type = get_edge_type(graph, each_node_id, right_node_id)

        if (left_edge_type[-1] == ">") and (right_edge_type[0] == "<"):
            definite_collider.add(each_node_id)
            continue 

        if (left_edge_type[-1] == "-") or (right_edge_type[0] == "-"):
            definite_non_collider.add(each_node_id)
            continue   

        if (left_edge_type[-1] == "∘") and (right_edge_type[0] == "∘"):
            if (graph[left_node_id, right_node_id] == 0) and (graph[right_node_id, left_node_id] == 0):
                definite_non_collider.add(each_node_id)
                continue

        is_definite_path = False

    
    return is_definite_path, definite_collider, definite_non_collider


def print_paths_with_edge_types(graph, paths, node_names=None):
    """
    打印路径及其精确边类型信息
    
    参数:
        graph: adjacency matrix of the PAG graph
        paths: 路径列表
        node_names: 可选，节点名称列表

    
    PAG边类型标记方案（与标准对应）:
        --> : A → B (G[A,B]=-1, G[B,A]=1)
        <-- : A ← B (G[A,B]=1, G[B,A]=-1)
        <-∘ : A ←∘ B (G[A,B]=2, G[B,A]=1)
        ∘-> : A ∘→ B (G[A,B]=1, G[B,A]=2)
        ∘─∘ : A ∘─∘ B (G[A,B]=2, G[B,A]=2)
        <─> : A ↔ B (G[A,B]=1, G[B,A]=1)
    """
    if node_names is None:
        node_names = [str(i) for i in range(graph.shape[0])]
    
    for i, path in enumerate(paths, 1):
        print(f"路径 {i}: ", end="")
        path_str = []
        edge_types = []
        
        for j in range(len(path)-1):
            from_node = path[j]
            to_node = path[j+1]
            
            edge_type = get_edge_type(graph, from_node, to_node)
            edge_types.append(edge_type)

            path_str.append(f"[{node_names[from_node]}]{edge_type}[{node_names[to_node]}]")
        
        # 打印带有节点名称的路径
        print(" → ".join([node_names[node] for node in path]))
        print("   边类型: " + " ".join(edge_types))
        print("   详细路径: " + " ".join(path_str))
        print()


def check_possibly_directed(graph, path):
    """
        判断是否是从左到右的有向路径
        :param g: PAG图
        :param path: 路径
        :return: bool
    """
    path_length = len(path)
    for i in range(path_length - 1):
        from_node = path[i]
        to_node = path[i + 1]
        edge_type = get_edge_type(graph, from_node, to_node)
        
        # 检查是否有向边
        if edge_type[0] == "<":
            return False 

    return True


def check_m_sep(conditioning_set, path, graph):
    """
    Check if a path is m-separated given a conditioning set in a PAG.
    
    Parameters:
    -----------
    conditioning_set : set
        The set of nodes we are conditioning on
    path : list
        A path represented as a list of node indices
    graph : numpy.ndarray
        The PAG represented as an adjacency matrix
    
    Returns:
    --------
    m_sep : bool or None
        True if the path is m-separated by the conditioning set
        False if the path is not m-separated
        None if the path status cannot be determined definitively
    """
    # Get the directed edges and compute the reachability matrix
    directed_graph = get_directed_graph(graph)
    reachability_matrix = linalg.expm(directed_graph)
    
    # Get the definite status of nodes in the path
    is_definite_path, colliders, non_colliders = get_definite_status_nodes(path, graph)
    
    # If we can't determine the definite status of all nodes in the path
    if not is_definite_path:
        return None
    
    # check the two conditions for m-connection.

    # Check condition 1: The conditioning set should not contain any non-collider
    contains_no_non_collider = conditioning_set.intersection(non_colliders) == set()
    
    # Check condition 2: For each collider, the conditioning set should contain 
    # either the collider or one of its descendants
    contains_all_collider_or_descendants = True
    for collider in colliders:
        # Get all descendants of the collider
        descendants = set(np.arange(reachability_matrix.shape[0])[reachability_matrix[:, collider] != 0])
        # Check if conditioning set contains either collider or any of its descendants
        if conditioning_set.intersection(descendants.union({collider})) == set():
            contains_all_collider_or_descendants = False
            break
    
    # A path is m-separated if either:
    # 1. It contains a non-collider that is in the conditioning set, or
    # 2. It contains a collider where neither the collider nor any of its descendants 
    #    are in the conditioning set
    m_sep = not (contains_no_non_collider and contains_all_collider_or_descendants)
    
    return m_sep


def possible_descendants(graph, nodes):
    """Possible directed reachability, including each starting node itself."""
    adjacency = get_possible_graph(graph)
    reached = set(nodes)
    pending = list(nodes)
    while pending:
        current = pending.pop()
        for child in np.flatnonzero(adjacency[:, current]):
            if int(child) not in reached:
                reached.add(int(child))
                pending.append(int(child))
    return reached


def get_bk(T_idx, Y_idx, N):
    """
    获取背景知识（时间顺序约束）

    Args:
        T_idx: 处理变量索引
        Y_idx: 结果变量索引
        N: 变量总数

    Returns:
        BackgroundKnowledge 对象
    """
    bk = BackgroundKnowledge()
    # 默认情况下，不添加特定的背景知识约束
    # 如果需要添加约束（如 T -> Y），可以使用：
    # bk.add_forbidden_relation(T_idx, Y_idx)  # 禁止 Y -> T
    # bk.add_required_relation(T_idx, Y_idx)    # 要求 T -> Y
    return bk


def is_directed_edge(graph: np.ndarray, i: int, j: int) -> bool:
    """
    判断是否存在有向边 i -> j。
    """
    return (graph[i, j] == -1) and (graph[j, i] == 1)


def is_bidirected_edge(graph: np.ndarray, i: int, j: int) -> bool:
    """
    判断是否存在双向边 i <-> j 
    """
    return (graph[i, j] == 1) and (graph[j, i] == 1)


def is_adjacent(graph: np.ndarray, i: int, j: int) -> bool:
    """判断节点 i 和 j 是否相邻（存在任何边）"""
    return graph[i, j] != 0 or graph[j, i] != 0


def is_parent(graph: np.ndarray, i: int, j: int) -> bool:
    """判断 i 是否是 j 的 parent（即存在 i -> j）"""
    return is_directed_edge(graph, i, j)


def is_star_arrow_edge(graph: np.ndarray, i: int, j: int) -> bool:
    """
    判断是否存在有向边 i *-> j。
    """
    return (graph[j,i] == 1)


def is_collider(graph: np.ndarray, a: int, c: int, b: int) -> bool:
    """判断 c 是否是 a 和 b 之间的 collider（a *-> c <-* b）"""
    return is_star_arrow_edge(graph, a, c) and is_star_arrow_edge(graph, b, c)


def is_visible(graph: np.ndarray, x: int, y: int) -> bool:
    """
    判断 PAG 中的有向边 X -> Y 是否为 Visible Edge。
    
    根据 Perkovic et al. (2018):
    X -> Y 是 visible 的，如果存在节点 V 不与 Y 相邻，且满足：
    1. V *-> X (存在指向 X 的边), 或
    2. 存在 collider path V *-> ... <-> X，且路径上所有非端点节点都是 Y 的 parent
    
    Args:
        graph: (n_nodes, n_nodes) 的邻接矩阵。约定与 causallearn 一致，并被
               get_edge_type 等函数沿用：graph[i, j] 编码边 i-j 在
               节点 i 一侧的端点:
               -1: tail 
               1: arrow 
               2: cirvle
               0: edge not exit
        x: 源节点 X 的索引
        y: 目标节点 Y 的索引
        
    Returns:
        bool: 如果是 visible edge 返回 True，否则返回 False
    """
    if not is_directed_edge(graph, x, y):
        print(f"节点 {x} 到 {y} 不存在有向边 X->Y")
        return False 
    
    n_nodes = graph.shape[0]
    
    # 条件 A: 寻找 V 使得 V -> X 且 V 与 Y 不相邻
    for v in range(n_nodes):
        if v == x or v == y:
            continue
            
        if is_star_arrow_edge(graph, v, x) and not is_adjacent(graph, v, y):
            return True
    
    # 条件 B: 寻找 collider path V <-> ... <-> X
    for v in range(n_nodes):
        if v == x or v == y or is_adjacent(graph, v, y):
            continue
            
        if _has_valid_collider_path(graph, v, x, y):
            return True
    
    return False


def _has_valid_collider_path(graph: np.ndarray, start: int, end: int, y: int) -> bool:
    """Find a collider path into end whose internal nodes are parents of y.

    The first edge may be start -> A, start o-> A or start <-> A;
    requiring every edge to be bidirected misses valid visibility witnesses.
    """
    def dfs(current: int, path: List[int]) -> bool:
        if current == end:
            return len(path) > 1 and is_star_arrow_edge(graph, path[-2], end)

        for neighbor in range(graph.shape[0]):
            if neighbor in path or not is_adjacent(graph, current, neighbor):
                continue
            # Appending a neighbor makes current an internal node. Check its
            # two arrowheads and parent relation before exploring further.
            if len(path) > 1 and not (
                is_collider(graph, path[-2], current, neighbor)
                and is_parent(graph, current, y)
            ):
                continue
            if dfs(neighbor, path + [neighbor]):
                return True
        return False

    return dfs(start, [start])


def GetMB(G, node_name, y_node=0):
    if isinstance(y_node, np.ndarray):
        try:
            y_node = y_node.item()  # 如果 y_node 是单个元素的数组，提取其值
        except:
            print("debug")
            print(y_node)

    mbset = set([y_node])
    d = G.shape[0]

    get_direct_set = lambda x: set(
        [idx for idx in range(d) if np.abs(G[x, idx]) + np.abs(G[idx, x]) > 0]
    )

    direct_set = get_direct_set(y_node)
    mbset = mbset.union(direct_set)

    for idx in direct_set:
        if G[idx, y_node] == -1:
            continue

        for each_secondary in get_direct_set(idx):
            if G[idx, each_secondary] == -1:
                continue
            mbset.add(each_secondary)

    return set([node_name[i] for i in mbset])

