import networkx as nx

def get_reference_path(start, targets, graph):

    if start not in graph.nodes or any(target not in graph.nodes for target in targets):
        raise ValueError("Either the source or target node is not present in the graph.")

    all_paths = []
    for target in targets:
        paths = nx.all_shortest_paths(graph, start, target)
        all_paths.extend(paths)

    reference_path = None
    reference_path_length = float('inf')
    for path in all_paths:
        if set(targets).issubset(path):
            # path_length = nx.shortest_path_length(graph, source=start, target=path[-1]) # Dijkstra algorithm
            path_length = nx.astar_path_length(graph, source=start, target=path[-1]) # A* algorithm
            if path_length < reference_path_length:
                reference_path = path
                reference_path_length = path_length

        else:
            reference_path = []
            for target in targets:
                reference_path.extend(nx.shortest_path(graph, start, target))

            reference_path = list(set(reference_path))

    reference_path_stack = reference_path[::-1]

    return reference_path_stack



def get_predecessors_within_k_hop(node, graph, k):
    predecessors = set()
    queue = [(node, 0)]

    while queue:
        current_node, hop = queue.pop(0)
        if hop > k-1:
            break
        for predecessor in graph.predecessors(current_node):
            predecessors.add(predecessor)
            queue.append((predecessor, hop + 1))

    return predecessors


def get_successors_within_1_hop(node, graph):
    successors = set()
    if node in graph:
        successors = graph[node]

    return successors




def Adaptive_Cognitive_Nevigation(target_nodes, knowledge_structure, learning_item, mastery, k_hop):


    reference_path = get_reference_path(learning_item, target_nodes, knowledge_structure.to_undirected())

    if learning_item == reference_path[-1]: # following reference
        if mastery:
            if len(reference_path)==1:
                return [reference_path[-1]]  # learning next
            else:
                reference_path.pop()  # mastered
                return [reference_path[-1]]  # learning next

        else:
            candidates = get_predecessors_within_k_hop(learning_item,knowledge_structure,k_hop)
            candidates = list(candidates)
            if not candidates:
                candidates.append(learning_item)
            return list(candidates)
    else:

        return [reference_path[-1]]


if __name__ == '__main__':

    # Demo
    G = nx.DiGraph()
    G.add_edges_from([(0,1),(0,2),(1,3),(2,4),(3,4),(2,8),(4,8),(5,4),(5,9),(6,7),(7,8),(8,9)])

    start_node = 4
    target_nodes = [6,9]

    reference_path = get_reference_path(start_node,target_nodes,G.to_undirected())

    print(reference_path)

    candidates, sp = Adaptive_Cognitive_Nevigation(target_nodes, G,4,0,1)

    print(candidates)
    print(sp)