from itertools import combinations
import networkx as nx
import utils
from time import perf_counter

# return degree of a group of nodes S
def group_degree(H, S):
    return len(set().union(*list(set(H.neighbors(i)) for i in S)) - set(S))

def most_degree_central_shortest_path_algorithm(G, s, t, prune=False):
    
    # find all shortest path lengths from s and t and create layers
    s_all = nx.single_source_shortest_path_length(G, s)
    t_all = nx.single_source_shortest_path_length(G, t)
    assert s_all[t] == t_all[s]  # assert that the distance from s to t is equal to distance from t to s
    d_st = s_all[t]

    # define dictionary of nodes with distance l to s and d_st-l to t (layers)
    layers = {layer: set(node for node in G if s_all[node] == layer and t_all[node] == d_st - layer) for layer in range(1, d_st)}
    # s - is the layer 0, and t is a layer d_st
    layers[0] = {s}
    layers[d_st] = {t}

    # define candidate neighbors for each node in each layer (where you can move from one layer to another)
    cand_neigh = {}
    for layer in range(d_st - 1):
        for node in layers[layer]:
            cand_neigh[node] = layers[layer + 1] & set(G.neighbors(node))
    # create a dictionary of path lists [key=len]
    path_list = {0: [tuple([s])]}
    for layer in range(1, d_st):
        # expand path list using nodes in layer 1
        path_list_new = [path + (neigh,) for path in path_list[layer - 1] for neigh in cand_neigh[path[-1]]]
        # if pruning option is selected
        if prune and layer >= 3:
            # compute path degrees
            path_deg = {path: group_degree(G, path) for path in path_list_new}
            # create the dictionary of paths with the same last three letters
            # and keep only one path with largest degree
            d_last_two = {}
            for path in path_list_new:
                # if there exists a path leave the path with maximum degree
                if path[-2:] in d_last_two:
                    if path_deg[path] >= path_deg[d_last_two[path[-2:]]]:
                        d_last_two[path[-2:]] = path
                else:
                    d_last_two[path[-2:]] = path
            # leave only paths with the largest degrees in each group
            path_list_new = [d_last_two[item] for item in d_last_two]
        path_list[layer] = path_list_new.copy()
    # add target node
    path_list[d_st] = [path + (t,) for path in path_list[d_st - 1]]

    path_deg = {path: group_degree(G, path) for path in path_list[d_st]}
    most_central_path = max(path_deg.items(), key=lambda x: x[1])

    return most_central_path[1], most_central_path[0], path_list

def MVP(G):
    spath_sol_a = []
    spath_neigh_a = []
    for s, t in combinations(G.nodes, 2):
        if (s, t) in G.edges or (t, s) in G.edges:
            continue
        spath_al_degree, spath_al_path, _ = most_degree_central_shortest_path_algorithm(G, s, t, prune=False)
        spath_sol_a.append(spath_al_path)
        spath_neigh_a.append(spath_al_degree)

    # get the shortest path with the largest neighbourhood
    spath_neigh_a_max_val = max(spath_neigh_a)
    spatha_sol = [spath_sol_a[idx] for (idx, val) in enumerate(spath_neigh_a) if val == spath_neigh_a_max_val]

    return spath_neigh_a_max_val, spatha_sol[0]

if __name__ == '__main__':

    G, G_nx = utils.read_saved_instance('./data/unweighted/undirected/watts-strogatz/100-ws-10/ws_100_4_10_0.csv', directed=False, w_max=10, seed=0, return_nxG=True)

    start = perf_counter()
    result = MVP(G_nx)
    end = perf_counter()
    elapsed = end - start
    print(elapsed)
    print(result)