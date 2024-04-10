# coding: utf-8

def mutil_cognitive_navigation(C, G, T, k): # C: current node, G: graph, T: target node, k: hop
    D_all = []
    for t in T:
        d = cognitive_navigation(C, G, t, k)
        D_all.append(d)

    D_all_set = list(set().union(*D_all))

    return D_all_set

def cognitive_navigation(C, G, T, k): # C: current node, G: graph, T: target node, k: hop
    D = set()
    Q = set()

    D.add(C)
    for successor in get_successors_within_k_minus_1_hop(C, G, k):
        D.add(successor)
    for predecessor in get_predecessors_within_k_hop(C, G, k):
        Q.add(predecessor)

    while Q:
        q = Q.pop()
        D.add(q)
        for successor in get_successors_within_1_hop(q, G):
            D.add(successor)

    D = remove_unreachable_targets(D, T, G)

    return D


def get_successors_within_k_minus_1_hop(node, graph, k):
    successors = set()
    queue = [(node, 0)]

    while queue:
        current_node, hop = queue.pop(0)
        if hop >= k-1:
            break
        for successor in graph[current_node]:
            successors.add(successor)
            queue.append((successor, hop + 1))

    return successors


def get_predecessors(node, graph):
    predecessors = []
    for key, value in graph.items():
        if node in value:
            predecessors.append(key)
    return predecessors


def get_predecessors_within_k_hop(node, graph, k):
    predecessors = set()
    queue = [(node, 0)]

    while queue:
        current_node, hop = queue.pop(0)
        if hop > k:
            break
        for predecessor in get_predecessors(current_node, graph):
            predecessors.add(predecessor)
            queue.append((predecessor, hop + 1))

    return predecessors


def get_successors_within_1_hop(node, graph):
    successors = set()
    if node in graph:
        successors = graph[node]

    return successors


def remove_unreachable_targets(nodes, target, graph):
    reachable_nodes = set()

    for node in nodes:
        if has_path_to_target(node, target, graph):
            reachable_nodes.add(node)

    return reachable_nodes


def has_path_to_target(node, target, graph):
    visited = set()
    stack = [node]

    while stack:
        current_node = stack.pop()
        if current_node == target:
            return True
        if current_node not in visited:
            visited.add(current_node)
            if current_node in graph:
                stack.extend(graph[current_node])

    return False







if __name__ == '__main__':
    C = 'E'
    G = {'C': ['A', 'B'], 'A': ['D'], 'B': ['D'], 'D': ['E', 'F'], 'E': ['G'], 'F': ['G']}
    T = ['G','F']
    k = 2

    print(mutil_cognitive_navigation(C, G, T, k))