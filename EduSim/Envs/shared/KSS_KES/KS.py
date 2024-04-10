# coding: utf-8

import networkx as nx


class KS(nx.DiGraph):
    def dump_id2idx(self, filename):
        with open(filename, "w") as wf:
            for node in self.nodes:
                print("%s,%s" % (node, node), file=wf)

    def dump_graph_edges(self, filename):
        with open(filename, "w") as wf:
            for edge in self.edges:
                print("%s,%s" % edge, file=wf)




def bfs(graph, mastery, pnode, hop, candidates, soft_candidates, visit_nodes=None, visit_threshold=1,
        allow_shortcut=True):  # pragma: no cover

    assert hop >= 0
    if visit_nodes and visit_nodes.get(pnode, 0) >= visit_threshold:
        return

    if allow_shortcut is False or mastery[pnode] < 0.5:
        candidates.add(pnode)
    else:
        soft_candidates.add(pnode)

    if hop == 0:
        return


    for node in list(graph.predecessors(pnode)):
        if allow_shortcut is False or mastery[node] < 0.5:
            bfs(
                graph=graph,
                mastery=mastery,
                pnode=node,
                hop=hop - 1,
                candidates=candidates,
                soft_candidates=soft_candidates,
                visit_nodes=visit_nodes,
                visit_threshold=visit_threshold,
                allow_shortcut=allow_shortcut,
            )

    for node in list(graph.successors(pnode)):
        if visit_nodes and visit_nodes.get(node, 0) >= visit_threshold:
            continue
        if allow_shortcut is False or mastery[node] < 0.5:
            candidates.add(node)
        else:
            soft_candidates.add(node)


def influence_control(graph, mastery, pnode, visit_nodes=None, visit_threshold=1, allow_shortcut=True, no_pre=None,
                      connected_graph=None, target=None, legal_candidates=None,
                      path_table=None) -> tuple:  # pragma: no cover
    """

    Parameters
    ----------
    graph: nx.Digraph
    mastery: list(float)
    pnode: None or int
    visit_nodes: None or dict
    visit_threshold: int
    allow_shortcut: bool
    no_pre: set
    connected_graph: dict
    target: set or list
    legal_candidates: set or None
    path_table: dict or None

    Returns
    -------

    """
    assert pnode is None or isinstance(pnode, int), pnode

    if mastery is None:
        allow_shortcut = False

    # select candidates
    candidates = []
    soft_candidates = []

    if allow_shortcut is True:

        if pnode is not None and mastery[pnode] >= 0.5:
            for candidate in list(graph.successors(pnode)):
                if visit_nodes and visit_nodes.get(candidate, 0) >= visit_threshold:
                    continue
                if mastery[candidate] < 0.5:
                    candidates.append(candidate)
                else:
                    soft_candidates.append(candidate)
            if candidates:
                return candidates, soft_candidates

        elif pnode is not None:
            _candidates = set()
            _soft_candidates = set()
            for node in list(graph.predecessors(pnode)):
                bfs(graph, mastery, node, 2, _candidates, _soft_candidates, visit_nodes, visit_threshold,
                    allow_shortcut)
            return list(_candidates) + [pnode], list(_soft_candidates)


        for node in graph.nodes:
            if visit_nodes and visit_nodes.get(node, 0) >= visit_threshold:

                continue

            if mastery[node] >= 0.5:

                soft_candidates.append(node)
                continue


            pre_nodes = list(graph.predecessors(node))
            for n in pre_nodes:
                pre_mastery = mastery[n]
                if pre_mastery < 0.5:
                    soft_candidates.append(node)
                    break
            else:
                candidates.append(node)
    else:

        candidates = set()
        soft_candidates = set()
        if pnode is not None:

            candidates = set(list(graph.successors(pnode)))

            if not graph.predecessors(pnode) or not graph.successors(pnode):

                candidates = set(no_pre)

            for node in list(graph.predecessors(pnode)):
                bfs(graph, mastery, node, 1, candidates, soft_candidates, visit_nodes, visit_threshold, allow_shortcut)


            if candidates:
                candidates.add(pnode)

            if visit_nodes:
                candidates -= set([node for node, count in visit_nodes.items() if count >= visit_threshold])

            candidates = list(candidates)

    if not candidates:

        candidates = list(graph.nodes)
        soft_candidates = list()

    if connected_graph is not None and pnode is not None:

        candidates = list(set(candidates) & connected_graph[pnode])

    if target is not None and legal_candidates is not None:
        assert target

        _candidates = set(candidates) - legal_candidates
        for candidate in _candidates:
            if candidate in legal_candidates:
                continue
            for t in target:
                if path_table is not None:
                    if t in path_table[candidate]:
                        legal_tag = True
                    else:
                        legal_tag = False
                else:
                    legal_tag = nx.has_path(graph, candidate, t)
                if legal_tag is True:
                    legal_candidates.add(candidate)
                    break
        candidates = set(candidates) & legal_candidates
        if not candidates:
            candidates = target

    return list(candidates), list(soft_candidates)
