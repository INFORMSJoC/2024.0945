import heapq
import utils
from time import perf_counter

def MostDegreeSingle(G, s, inf=1e9):
    '''
    Function to solve the problem of finding the shortest path with largest the neighbourhood
    for an unweighted graph G from a starting node s (Algorithm 1).
    
    Inputs:
    G: Graph as dict of dict of weights. The weights will be ignored.
        If w_{ij} is the weight of the edge from i to j, then G[i][j] = w_{ij}.
        Note that if vertex i has no edges, it must be included in the dictionary.
        Example input: {0: {1: 2, 2: 1, 3: 1, 4: 1}, 
                        1: {0: 2, 3: 1}, 
                        2: {0: 1, 4: 3}, 
                        3: {0: 1, 1: 1}, 
                        4: {0: 1, 2: 3}}
    s: starting node. Must be in G.
    inf: Large number for non-existent paths.
    
    Returns:
    dists: dict
        where dist[i] indicates the shortest path from s to vertex i
    path: dict
        where path[i] indicates the most degree central shortest paths from s to i
    neigh: dict
        where neigh[i] indicates the neighbours of the most degree central shortest paths from s to i
    '''
    
    dists = {v: inf for v in G}
    path = {v: [] for v in G}
    prev = {v: set() for v in G}
    neigh = {v: set() for v in G}

    neigh[s] = set(G[s].keys())
    dists[s] = 0
    path[s].append(s)
    Q = [(0, s)]
    
    while Q:
        ## Consider current minimum distance as a starting point
        current_distance, u = heapq.heappop(Q)
        
        if current_distance > dists[u]:
            continue

        for v, weight in G[u].items():
            
            d_new = current_distance + 1

            if d_new == 1:
                heapq.heappush(Q, (d_new, v))
                dists[v] = 1
                prev[v].add(u)
                path[v] = [u, v]

                neigh[v] = neigh[u].union(G[v].keys()) - set(path[v])

            ## Add to path if its shorter than all previous paths
            elif d_new < dists[v]:
                heapq.heappush(Q, (d_new, v))
                dists[v] = d_new
                prev[v].add(u)
                for w in prev[u]:
                    new_path = path[w] + [u, v]
                    new_neigh = neigh[w].union(G[u].keys()).union(G[v].keys()) - set(new_path)
                    if len(new_neigh) > len(neigh[v]):
                        neigh[v] = new_neigh
                        path[v] = new_path
                
            ## If there is a tie, compare the neighbours to see which path is selected
            elif abs(d_new - dists[v]) < 1e-6:

                prev[v].add(u)

                for w in prev[u]:
                    new_path = path[w] + [u, v]
                    new_neigh = neigh[w].union(G[u].keys()).union(G[v].keys()) - set(new_path)
                    if len(new_neigh) > len(neigh[v]):
                        neigh[v] = new_neigh
                        path[v] = new_path
    return dists, path, neigh


def MostDegreeGraph(G, inf=1e9):
    '''
    Function to solve the problem of finding the finding the shortest path with largest the neighbourhood
    for a unweighted graph G.
    
    Inputs:
    G: Graph as dict of dict of weights. e weights will be ignored.
        If w_{ij} is the weight of the edge from i to j, then G[i][j] = w_{ij}.
        Note that if vertex i has no edges, it must be included in the dictionary.
        Example input: {0: {1: 2, 2: 1, 3: 1, 4: 1}, 
                        1: {0: 2, 3: 1}, 
                        2: {0: 1, 4: 3}, 
                        3: {0: 1, 1: 1}, 
                        4: {0: 1, 2: 3}}
    inf: Large number for non-existent paths.
    
    Returns:
    solution: dict
        dictionary with keys:
            - 'centrality': the centrality of the most degree-central shortest path
            - 'neighborhood': the neighbourhood of the most degree-central shortest path
            - 'path': the most degree-central shortest path
            - 'dist': the length of the most degree-central shortest path
    diam_info: dict
        similar to `solution`, but with data on the most degree-central longest shortest path in G.
    '''

    max_neigh = {}
    max_neigh_path = []
    path_dist = inf
    path_n_neighbours = 0

    diam_max_neigh = {}
    diam_max_neigh_path = []
    diam_dist = 0
    diam_n_neighbours = 0

    for s in G:
        dists, paths, neighs = MostDegreeSingle(G, s)
        
        for t in G:
            dist = dists[t]
            if dist == inf: # Ignore (s, t) that aren't connected
                continue
            path = paths[t]
            neighbours = neighs[t]
            n_neighbours = len(neighbours)

            # More degree-central shortest path
            if n_neighbours > path_n_neighbours:
                max_neigh = neighbours
                max_neigh_path = path
                path_dist = dist
                path_n_neighbours = n_neighbours
                
            # Keep the shortest
            elif n_neighbours == path_n_neighbours:                
                if path_dist > dist:
                    max_neigh = neighbours
                    max_neigh_path = path
                    path_dist = dist

            # Get the longest shortest path
            if dist > diam_dist:
                diam_max_neigh = neighbours
                diam_max_neigh_path = path
                diam_dist = dist
                diam_n_neighbours = n_neighbours
            
            # Get the most degree central
            elif dist == diam_dist:                
                if n_neighbours > diam_n_neighbours:
                    diam_max_neigh = neighbours
                    diam_max_neigh_path = path
                    diam_n_neighbours = n_neighbours

    solution = {
        'centrality': path_n_neighbours,
        'neighborhood': max_neigh, 
        'path': max_neigh_path, 
        'dist': path_dist
    }
    
    diam_info = {
        'centrality': diam_n_neighbours,
        'neighborhood': diam_max_neigh, 
        'path': diam_max_neigh_path, 
        'dist': diam_dist
    }

    return solution, diam_info

def MostDegreeSingleWeighted(G_aug, G_orig, s, inf=1e9):
    '''
    Function to solve the problem of finding the shortest path with largest the neighbourhood
    for a weighted graph G from a starting node s (extension of Algorithm 1).
    
    Inputs:
    G_aug: The augmented graph as dict of dict of weights.
        Use utils.aug_weighted_to_unweighted to generate this graph from G. 
        It is important that `G.keys().union(G_orig.keys()) == G_orig.keys()`
    G_orig: Graph as dict of dict of weights.
        If w_{ij} is the weight of the edge from i to j, then G[i][j] = w_{ij}.
        Note that if vertex i has no edges, it must be included in the dictionary.
        Example input: {0: {1: 2, 2: 1, 3: 1, 4: 1}, 
                        1: {0: 2, 3: 1}, 
                        2: {0: 1, 4: 3}, 
                        3: {0: 1, 1: 1}, 
                        4: {0: 1, 2: 3}}
    s: starting node. Must be in G_aug and G_orig.
    
    Returns:
    dists: dict
        where dist[i] indicates the shortest path from s to vertex i
    path: dict
        where path[i] indicates the most degree central shortest paths from s to i
    neigh: dict
        where neigh[i] indicates the neighbours of the most degree central shortest paths from s to i
    '''
    
    dists = {v: inf for v in G_aug}
    path = {v: [] for v in G_aug}    
    original_path = {v: [] for v in G_aug}
    prev = {v: set() for v in G_aug}
    neigh = {v: set() for v in G_aug}


    neigh[s] = set(G_orig[s].keys())
    dists[s] = 0
    path[s].append(s)
    original_path[s].append(s)
    Q = [(0, s)]
    
    while Q:
        ## Consider current minimum distance as a starting point
        current_distance, u = heapq.heappop(Q)
        
        if current_distance > dists[u]:
            continue

        for v, weight in G_aug[u].items():
            
            d_new = current_distance + 1

            if d_new == 1:
                heapq.heappush(Q, (d_new, v))
                dists[v] = 1
                prev[v].add(u)
                path[v] = [u, v]
                
                if v in G_orig:
                    original_path[v] = [u, v]
                    neigh[v] = neigh[u].union(G_orig[v].keys()) - set(path[v])
                else:
                    original_path[v] = [u, v]
                    neigh[v] = neigh[u]

            ## Add to path if its shorter than all previous paths
            elif d_new < dists[v]:
                heapq.heappush(Q, (d_new, v))
                dists[v] = d_new
                prev[v].add(u)
                if u in G_orig:
                    for w in prev[u]:
                        new_path = path[w] + [u, v]
                        if v in G_orig:
                            new_original_path = original_path[w] + [u, v]
                            new_neigh = neigh[w].union(G_orig[u].keys()).union(G_orig[v].keys()) - set(new_original_path)
                        else:
                            new_original_path = original_path[w] + [u]
                            new_neigh = neigh[w].union(G_orig[u].keys()) - set(new_original_path)

                        if len(new_neigh) > len(neigh[v]):
                            neigh[v] = new_neigh
                            path[v] = new_path
                            original_path[v] = new_original_path

                else: # there is only one prev[w] possible
                    path[v] = path[u] + [v]                    
                    if v in G_orig:
                        original_path[v] = original_path[u] + [v]
                        neigh[v] = neigh[u].union(G_orig[v].keys()) - set(path[v])
                    else:
                        original_path[v] = original_path[u]
                        neigh[v] = neigh[u]
                
            ## If there is a tie, compare the neighbours to see which path is selected
            elif (abs(d_new - dists[v]) < 1e-6):

                prev[v].add(u)

                for w in prev[u]:
                    new_path = path[w] + [u, v]
                    if u in G_orig:
                        if v in G_orig:
                            new_original_path = original_path[w] + [u, v]
                            new_neigh = neigh[w].union(G_orig[u].keys()).union(G_orig[v].keys()) - set(new_original_path)
                        else:
                            new_original_path = original_path[w] + [u]
                            new_neigh = neigh[w].union(G_orig[u].keys()) - set(new_original_path)
                    else:
                        if v in G_orig:
                            new_original_path = original_path[w] + [v]
                            new_neigh = neigh[w].union(G_orig[v].keys()) - set(new_original_path)
                        else:
                            new_original_path = original_path[w]
                            new_neigh = neigh[w] - set(new_original_path)

                    if len(new_neigh) > len(neigh[v]):
                        neigh[v] = new_neigh
                        path[v] = new_path
                        original_path[v] = original_path[w]
    return dists, path, neigh

def MostDegreeGraphWeighted(G_aug, G_orig, inf=1e9):
    '''
    Function to solve the problem of finding the shortest path with largest the neighbourhood
    for a weighted graph G_orig and its augmented version G_aug
    
    Inputs:
    G_aug: Graph as dict of dict of weights. The weights will be ignored.
        If w_{ij} is the weight of the edge from i to j, then G_aug[i][j] = w_{ij}.
        Note that if vertex i has no edges, it must be included in the dictionary.
        Example input: {0: {1: 2, 2: 1, 3: 1, 4: 1}, 
                        1: {0: 2, 3: 1}, 
                        2: {0: 1, 4: 3}, 
                        3: {0: 1, 1: 1}, 
                        4: {0: 1, 2: 3}}
    inf: Large number for non-existent paths.
    
    Returns:
    solution: dict
        dictionary with keys:
            - 'centrality': the centrality of the most degree-central shortest path
            - 'neighborhood': the neighbourhood of the most degree-central shortest path
            - 'path': the most degree-central shortest path
            - 'st': the start and end nodes of the most degree-central shortest path
            - 'dist': the length of the most degree-central shortest path

    diam_info: dict
        similar to `solution`, but with data on the most degree-central longest shortest path in G.
    '''

    max_neigh = {}
    max_neigh_path = []
    path_dist = inf
    path_n_neighbours = 0

    diam_max_neigh = {}
    diam_max_neigh_path = []
    diam_dist = 0
    diam_n_neighbours = 0

    original_vertices = G_orig.keys()

    for s in original_vertices:
        dists, paths, neighs = MostDegreeSingleWeighted(G_aug, G_orig, s)
        
        for t in original_vertices:
            dist = dists[t]
            if dist == inf: # Ignore (s, t) that aren't connected
                continue
            path = paths[t]
            neighbours = neighs[t]
            n_neighbours = len(neighbours)

            # More degree-central shortest path
            if n_neighbours > path_n_neighbours:
                max_neigh = neighbours
                max_neigh_path = path
                path_dist = dist
                path_n_neighbours = n_neighbours
                
            # Keep the shortest
            elif n_neighbours == path_n_neighbours:                
                if path_dist > dist:
                    max_neigh = neighbours
                    max_neigh_path = path
                    path_dist = dist

            # Get the longest shortest path
            if dist > diam_dist:
                diam_max_neigh = neighbours
                diam_max_neigh_path = path
                diam_dist = dist
                diam_n_neighbours = n_neighbours
            
            # Get the most degree central
            elif dist == diam_dist:                
                if n_neighbours > diam_n_neighbours:
                    diam_max_neigh = neighbours
                    diam_max_neigh_path = path
                    diam_n_neighbours = n_neighbours

    max_neigh_path = [i for i in max_neigh_path if i in G_orig]
    diam_max_neigh_path = [i for i in diam_max_neigh_path if i in G_orig]

    solution = {
        'centrality': path_n_neighbours,
        'neighborhood': max_neigh, 
        'path': max_neigh_path,
        'dist': path_dist
    }
    
    diam_info = {
        'centrality': diam_n_neighbours,
        'neighborhood': diam_max_neigh, 
        'path': diam_max_neigh_path,
        'dist': diam_dist
    }

    return solution, diam_info

if __name__ == '__main__':

    G, G_nx = utils.read_saved_instance('./data/unweighted/undirected/watts-strogatz/100-ws-10/ws_100_4_10_0.csv', directed=False, w_max=10, seed=0, return_nxG=True)

    start = perf_counter()
    result = MostDegreeGraph(G)
    end = perf_counter()
    elapsed = end - start
    print(elapsed)
    print(result)

    G_orig, G_nx_orig = utils.read_saved_instance('./data/weighted/directed/real-instances/copenhagen-calls-directed.csv', directed=True, return_nxG=True, to_string=True)
    G_aug, G_aug_nx = utils.aug_weighted_to_unweighted('./data/weighted/directed/real-instances/copenhagen-calls-directed.csv', directed=True, return_nxG=True)

    n_nodes = G_nx_orig.number_of_nodes()
    n_edges = G_nx_orig.number_of_edges()

    start_time = perf_counter()
    result = MostDegreeGraphWeighted(G_aug, G_orig)
    end_time = perf_counter()
    elapsed = end - start
    print(elapsed)
    print(result)