import networkx as nx
import numpy as np
from collections import deque
import heapq
from time import perf_counter
from utils import *


def check_in_queue(Q, x):
    for i in range(len(Q)):
        if x == Q[i][1]:
            return True, i
    return False


def MostBetweennessGraph(G, inf=1e9):
    '''
    Function to solve the problem of finding the finding the most betweenness shortest path for all (s, t) pairs in an unweighted graph G (Algorithm 2).
    
    Inputs:
    G: Graph as dict of dict of weights. The weights will be ignored.
        If w_{ij} is the weight of the edge from i to j, then G[i][j] = w_{ij}.
        Note that if vertex i has no edges, it must be included in the dictionary.
        Example input: {0: {1: 2, 2: 1, 3: 1, 4: 1}, 
                        1: {0: 2, 3: 1}, 
                        2: {0: 1, 4: 3}, 
                        3: {0: 1, 1: 1}, 
                        4: {0: 1, 2: 3}}
    inf: Large number for non-existent paths.

    Returns:
    results: dict of dicts
        dict with keys sn for each source node s. Each results[sn] contains dictionaries with the following keys
            - `results[sn]['P']`: dict with keys t where each value is the most betweenness-central shortest path from sn to t
            - `results[sn]['c']`: dict with keys t where each value is the betweenness centrality of the most betweenness-central shortest path from sn to t
    pp_time: float
        time to complete preprocessing
    it_count: int
        number of iterations group betweenness centrality is computed
    '''
    pp_start = perf_counter()

    # Preprocessing
    ## Store all results of path counting
    preprocess = {}

    single_node_centrality = {v: 0 for v in G}

    # Initialisation
    for s in G:
        pred = {w: [] for w in G}
        dist = {t: inf for t in G}
        sigma = {t: 0 for t in G}

        dist[s] = 0
        sigma[s] = 1

        Q = deque([s])
        S = []

        while Q:
            v = Q.popleft()
            S.append(v)

            for w in G[v]:
                ### Path discovery
                if dist[w] == inf:
                    dist[w] = dist[v] + 1
                    Q.append(w)

                ### Path counting
                if dist[w] == dist[v] + 1:
                    sigma[w] = sigma[w] + sigma[v]
                    pred[w].append(v)

        preprocess[s] = {'S': S.copy(), 'pred': pred, 'sigma': sigma.copy()}

        ### Get singleton centrality
        delta = {v: 0 for v in G}
        while S:
            w = S.pop()
            for v in pred[w]:
                delta[v] = delta[v] + (1 + delta[w])
            if w != s:
                single_node_centrality[w] = single_node_centrality[w] + sigma[w] * delta[w]

    pp_end = perf_counter()

    pp_time = pp_end - pp_start

    it_count = 0

    # Finding most betweenness shortest path for starting node sn
    results = {}
    for sn in G:
        d = {v: inf for v in G}
        P = {v: [] for v in G}
        c = {v: 0 for v in G}

        P[sn] = [sn]
        d[sn] = 0
        c[sn] = single_node_centrality[sn]

        Q = deque([(0, sn)])

        while Q:
            val = Q.popleft()
            d_u, u = val

            for v in G[u]:
                d_new = d_u + 1

                if d_new < d[v]:
                    Q.append((d_new, v))
                    d[v] = d_new
                    P[v] = P[u] + [v]

                    ### Compute centrality
                    c_new = 0

                    for s in G:
                        S = preprocess[s]['S'].copy()
                        pred = preprocess[s]['pred']
                        sigma = preprocess[s]['sigma']
                        delta = {v: 0 for v in G}
                        while S:
                            w = S.pop()
                            for v_c in pred[w]:
                                if w in P[v]:
                                    delta[v_c] = delta[v_c] + 1
                                else:
                                    delta[v_c] = delta[v_c] + (1 + delta[w])
                            if w in P[v] and w != s:
                                c_new = c_new + sigma[w] * delta[w]

                        it_count += 1

                    c[v] = c_new

                elif d_new == d[v]:
                    P_new = P[u] + [v]

                    ### Compute centrality
                    c_new = 0

                    for s in G:
                        S = preprocess[s]['S'].copy()
                        pred = preprocess[s]['pred']
                        sigma = preprocess[s]['sigma']

                        delta = {v: 0 for v in G}
                        while S:
                            w = S.pop()
                            for v_c in pred[w]:
                                if w in P_new:
                                    delta[v_c] = delta[v_c] + 1
                                else:
                                    delta[v_c] = delta[v_c] + (1 + delta[w])
                            if w in P_new and w != s:
                                c_new = c_new + sigma[w] * delta[w]

                        it_count += 1

                    if c_new > c[v]:
                        P[v] = P_new
                        c[v] = c_new

        results[sn] = {'P': P, 'c': c}

    return results, pp_time, it_count


def MostBetweennessWeightedGraph(G, inf=1e9):
    '''
    Function to solve the problem of finding the finding the most betweenness shortest path for all (s, t) pairs in a weighted graph G (Algorithm 2).
    
    Inputs:
    G: Graph as dict of dict of weights.
        If w_{ij} is the weight of the edge from i to j, then G[i][j] = w_{ij}.
        Note that if vertex i has no edges, it must be included in the dictionary.
        Example input: {0: {1: 2, 2: 1, 3: 1, 4: 1}, 
                        1: {0: 2, 3: 1}, 
                        2: {0: 1, 4: 3}, 
                        3: {0: 1, 1: 1}, 
                        4: {0: 1, 2: 3}}
    inf: Large number for non-existent paths.

    Returns:
    results: dict of dicts
        dict with keys sn for each source node s. Each results[sn] contains dictionaries with the following keys
            - 'results[sn]['P']: dict with keys t where each value is the most betweenness-central shortest path from sn to t
            - 'results[sn]['c']: dict with keys t where each value is the betweenness centrality of the most betweenness-central shortest path from sn to t
    pp_time: float
        time to complete preprocessing
    it_count: int
        number of iterations group betweenness centrality is computed
    '''
    pp_start = perf_counter()

    # Preprocessing
    ## Compute all sigma values

    ### Store all results
    preprocess = {}

    single_node_centrality = {v: 0 for v in G}

    # Initialisation
    for s in G:
        pred = {w: [] for w in G}
        dist = {t: inf for t in G}
        sigma = {t: 0 for t in G}

        dist[s] = 0
        sigma[s] = 1

        Q = [(0, s)]
        S = []

        while Q:
            d_v, v = heapq.heappop(Q)
            S.append(v)

            for w in G[v]:
                ### Path discovery
                if (d_new := dist[v] + G[v][w]) < dist[w] - 1e-6:
                    dist[w] = d_new
                    if (check_result := check_in_queue(Q, w)) == False:
                        heapq.heappush(Q, (d_new, w))
                    else:  ### Update priority queue
                        del Q[check_result[1]]
                        heapq.heappush(Q, (d_new, w))
                    sigma[w] = 0
                    pred[w] = []

                ### Path counting
                if abs(dist[w] - (dist[v] + G[v][w])) < 1e-6:
                    sigma[w] = sigma[w] + sigma[v]
                    pred[w].append(v)

        preprocess[s] = {'S': S.copy(), 'pred': pred, 'sigma': sigma.copy()}

        ### Get singleton centrality
        delta = {v: 0 for v in G}
        while S:
            w = S.pop()
            for v in pred[w]:
                delta[v] = delta[v] + (1 + delta[w])
            if w != s:
                single_node_centrality[w] = single_node_centrality[w] + sigma[w] * delta[w]

    pp_end = perf_counter()
    pp_time = pp_end - pp_start

    it_count = 0

    # Finding most betweenness shortest path for starting node sn
    results = {}
    for sn in G:
        d = {v: inf for v in G}
        P = {v: [] for v in G}
        c = {v: 0 for v in G}

        P[sn] = [sn]
        d[sn] = 0
        c[sn] = single_node_centrality[sn]

        Q = [(0, sn)]

        while Q:
            d_u, u = heapq.heappop(Q)
            for v in G[u]:
                d_new = d_u + G[u][v]

                if d_new < d[v] - 1e-6:
                    if (check_result := check_in_queue(Q, v)) == False:
                        heapq.heappush(Q, (d_new, v))
                    else:  ### Update priority queue
                        del Q[check_result[1]]
                        heapq.heappush(Q, (d_new, v))
                    d[v] = d_new
                    P[v] = P[u] + [v]

                    ### Compute centrality
                    c_new = 0

                    for s in G:
                        S = preprocess[s]['S'].copy()
                        pred = preprocess[s]['pred']
                        sigma = preprocess[s]['sigma']
                        delta = {v: 0 for v in G}
                        while S:
                            w = S.pop()
                            for v_c in pred[w]:
                                if w in P[v]:
                                    delta[v_c] = delta[v_c] + 1
                                else:
                                    delta[v_c] = delta[v_c] + (1 + delta[w])
                            if w in P[v] and w != s:
                                c_new = c_new + sigma[w] * delta[w]

                        it_count += 1

                    c[v] = c_new

                elif abs(d_new - d[v]) < 1e-6:
                    P_new = P[u] + [v]

                    ### Compute centrality
                    c_new = 0

                    for s in G:
                        S = preprocess[s]['S'].copy()
                        pred = preprocess[s]['pred']
                        sigma = preprocess[s]['sigma']

                        delta = {v: 0 for v in G}
                        while S:
                            w = S.pop()
                            for v_c in pred[w]:
                                if w in P_new:
                                    delta[v_c] = delta[v_c] + 1
                                else:
                                    delta[v_c] = delta[v_c] + (1 + delta[w])
                            if w in P_new and w != s:
                                c_new = c_new + sigma[w] * delta[w]

                        it_count += 1

                    if c_new > c[v]:
                        P[v] = P_new
                        c[v] = c_new

        results[sn] = {'P': P, 'c': c, 'd': d}

    return results, pp_time, it_count


def MostBetweennessGraphSolver(G, weighted=False, inf=1e9):
    '''
    Function to solve the problem of finding the finding the most betweenness shortest path in graph G (Algorithm 2).
    
    Inputs:
    G: Graph as dict of dict of weights. The weights will be ignored.
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
            - 'centrality': the centrality of the most betweenness-central shortest path
            - 'path': the most betweenness-central shortest path
            - 'length': the length of the most betweenness-central shortest path
    diam_info: dict
        similar to `solution`, but with data on the most betweenness-central longest shortest path in G.
    pp_time: float
        time to complete preprocessing
    it_count: int
        number of iterations group betweenness centrality is computed
    '''

    if weighted:
        results, pp_time, it_count = MostBetweennessWeightedGraph(G)
    else:
        results, pp_time, it_count = MostBetweennessGraph(G)

    sp_score = 0
    sp_path = []
    sp_length = inf

    diam_score = 0
    diam_path = []
    diam_length = 0

    for s in G:
        res_paths = results[s]['P']
        res_cent = results[s]['c']
        for t in res_paths:
            if weighted:
                res_length = results[s]['d'][t]
                if res_length == inf:
                    continue
            else:
                res_length = len(res_paths[t])

            if res_cent[t] > sp_score:
                sp_score = res_cent[t]
                sp_path = res_paths[t]
                sp_length = res_length
            elif res_cent[t] == sp_score:
                if res_length < sp_length:
                    sp_path = res_paths[t]
                    sp_length = res_length

            if res_length > diam_length:
                diam_length = res_length
                diam_path = res_paths[t]
                diam_score = res_cent[t]
            elif res_length == diam_length:
                if res_cent[t] > diam_score:
                    diam_path = res_paths[t]
                    diam_score = res_cent[t]

    if not weighted:
        sp_length -= 1
        diam_length -= 1

    solution = {'centrality': sp_score, 'path': sp_path, 'length': sp_length}
    diam_info = {'centrality': diam_score, 'path': diam_path, 'length': diam_length}

    return (solution, diam_info, pp_time, it_count)


if __name__ == '__main__':
    G, G_nx = read_saved_instance('./data/unweighted/undirected/watts-strogatz/100-ws-10/ws_100_4_10_0.csv', directed=False, w_max=10, seed=0, return_nxG=True)

    start = perf_counter()
    result = MostBetweennessGraphSolver(G, weighted=False)
    end = perf_counter()
    elapsed = end - start
    print(elapsed)
    print(result)