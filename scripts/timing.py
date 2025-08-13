import sys
import os
import pandas as pd
from time import perf_counter
from datetime import datetime
import multiprocessing as mp

sys.path.append('./src')
import utils
import Degree
import Betweenness
import MVP

use_multiprocessing = True
n_jobs = 10

def DegTime(graph_path):
    path_split = graph_path.split('/')
    graph_group = path_split[-2]
    graph_id = path_split[-1][:-4]
    weighted = path_split[2] == 'weighted'
    directed = path_split[3] == 'directed'

    G, Gnx = utils.read_saved_instance(graph_path, directed=directed, return_nxG=True)

    n_nodes = Gnx.number_of_nodes()
    n_edges = Gnx.number_of_edges()

    start_time = perf_counter()
    solution, diam_info = Degree.MostDegreeGraph(G)
    end_time = perf_counter()

    time_taken = end_time - start_time

    path_length = solution['dist']
    path_centrality = solution['centrality']
    path_nodes_traversed = len(solution['path'])
    path = solution['path']

    # Diameter info
    diam = diam_info['dist']
    diam_centrality = diam_info['centrality']
    diam_nodes_traversed = len(diam_info['path'])
    diam_path = diam_info['path']

    sol = [graph_group, graph_id, False, n_nodes, n_edges, weighted, directed, time_taken, path_centrality, path_length, path_nodes_traversed, path, diam_centrality, diam, diam_nodes_traversed, diam_path]
   
    print(sol)

    return sol

def DegWeightedTime(graph_path):
    path_split = graph_path.split('/')
    graph_group = path_split[-2]
    graph_id = path_split[-1][:-4]
    weighted = path_split[2] == 'weighted'
    directed = path_split[3] == 'directed'

    G_orig, G_nx_orig = utils.read_saved_instance(graph_path, directed=directed, return_nxG=True, to_string=True)
    G, Gnx = utils.aug_weighted_to_unweighted(graph_path, directed=False, return_nxG=True)

    n_nodes = G_nx_orig.number_of_nodes()
    n_edges = G_nx_orig.number_of_edges()

    start_time = perf_counter()
    solution, diam_info = Degree.MostDegreeGraphWeighted(G, G_orig)
    end_time = perf_counter()

    time_taken = end_time - start_time

    path_length = solution['dist']
    path_centrality = solution['centrality']
    path_nodes_traversed = len(solution['path'])
    path = solution['path']

    # Diameter info
    diam = diam_info['dist']
    diam_centrality = diam_info['centrality']
    diam_nodes_traversed = len(diam_info['path'])
    diam_path = diam_info['path']

    sol = [graph_group, graph_id, True, n_nodes, n_edges, weighted, directed, time_taken, path_centrality, path_length, path_nodes_traversed, path, diam_centrality, diam, diam_nodes_traversed, diam_path]
   
    print(sol)

    return sol

def MVPTime(graph_path):
    path_split = graph_path.split('/')
    graph_group = path_split[-2]
    graph_id = path_split[-1][:-4]
    weighted = path_split[2] == 'weighted'
    directed = path_split[3] == 'directed'

    G, Gnx = utils.read_saved_instance(graph_path, directed=directed, return_nxG=True)

    n_nodes = Gnx.number_of_nodes()
    n_edges = Gnx.number_of_edges()

    start_time = perf_counter()
    path_centrality, path = MVP.MVP(Gnx)
    end_time = perf_counter()

    path_length = len(path) - 1
    
    time_taken = end_time - start_time

    sol = [graph_group, graph_id, n_nodes, n_edges, weighted, directed, time_taken, path_centrality, path_length, path]
   
    print(sol)

    return sol


def BetweennessTime(graph_path):
    path_split = graph_path.split('/')
    graph_group = path_split[-2]
    graph_id = path_split[-1][:-4]
    weighted = path_split[2] == 'weighted'
    directed = path_split[3] == 'directed'

    G, G_nx = utils.read_saved_instance(graph_path, directed=directed, return_nxG=True)

    n_nodes = G_nx.number_of_nodes()
    n_edges = G_nx.number_of_edges()

    start_time = perf_counter()
    sp_info, diam_info, pp_time, it_count = Betweenness.MostBetweennessGraphSolver(G, weighted=weighted)
    end_time = perf_counter()

    time_taken = end_time - start_time

    path_centrality = sp_info['centrality']
    path_length = sp_info['length']
    path = sp_info['path']
    path_nodes_traversed = len(sp_info['path'])

    # Diameter info

    diam_centrality = diam_info['centrality']
    diam = diam_info['length']
    diam_path = diam_info['path']
    diam_nodes_traversed = len(diam_info['path'])

    sol = [graph_group, graph_id, n_nodes, n_edges, weighted, directed, time_taken, path_centrality, path_length, path_nodes_traversed, path, diam_centrality, diam, diam_nodes_traversed, diam_path, pp_time, it_count]

    print(sol)

    return sol

if __name__ == '__main__':

    ## Unweighted degree timing test

    udeg_paths_dict = {
        './data/unweighted/undirected/barabasi_albert/100-barabasi_albert/': os.listdir('./data/unweighted/undirected/barabasi_albert/100-barabasi_albert/'),
        './data/unweighted/undirected/watts-strogatz/100-ws-10/': os.listdir('./data/unweighted/undirected/watts-strogatz/100-ws-10/'),
        './data/unweighted/undirected/watts-strogatz/100-ws-20/': os.listdir('./data/unweighted/undirected/watts-strogatz/100-ws-20/'),
        './data/unweighted/undirected/barabasi_albert/500-barabasi_albert/': os.listdir('./data/unweighted/undirected/barabasi_albert/500-barabasi_albert/'),
        './data/unweighted/undirected/watts-strogatz/500-ws-10/': os.listdir('./data/unweighted/undirected/watts-strogatz/500-ws-10/'),
        './data/unweighted/undirected/watts-strogatz/500-ws-20/': os.listdir('./data/unweighted/undirected/watts-strogatz/500-ws-20/'),
        './data/unweighted/undirected/barabasi_albert/1000-barabasi_albert/': os.listdir('./data/unweighted/undirected/barabasi_albert/1000-barabasi_albert/'),
        './data/unweighted/undirected/watts-strogatz/1000-ws-10/': os.listdir('./data/unweighted/undirected/watts-strogatz/1000-ws-10/'),
        './data/unweighted/undirected/watts-strogatz/1000-ws-20/': os.listdir('./data/unweighted/undirected/watts-strogatz/1000-ws-20/'),
        './data/unweighted/undirected/barabasi_albert/5000-barabasi_albert/': os.listdir('./data/unweighted/undirected/barabasi_albert/5000-barabasi_albert/'),
        './data/unweighted/undirected/watts-strogatz/5000-ws-10/': os.listdir('./data/unweighted/undirected/watts-strogatz/5000-ws-10/'),
        './data/unweighted/undirected/watts-strogatz/5000-ws-20/': os.listdir('./data/unweighted/undirected/watts-strogatz/5000-ws-20/'),
        './data/unweighted/undirected/barabasi_albert/1000-barabasi_albert/': os.listdir('./data/unweighted/undirected/barabasi_albert/1000-barabasi_albert/'),
        './data/unweighted/undirected/watts-strogatz/1000-ws-10/': os.listdir('./data/unweighted/undirected/watts-strogatz/10000-ws-10/'),
        './data/unweighted/undirected/watts-strogatz/1000-ws-20/': os.listdir('./data/unweighted/undirected/watts-strogatz/10000-ws-20/'),
        './data/unweighted/directed/real-instances/': os.listdir('./data/unweighted/directed/real-instances/'),
        './data/unweighted/undirected/real-instances/': os.listdir('./data/unweighted/undirected/real-instances/'),
    }

    paths = []

    for directory in udeg_paths_dict:
        for graph_id in udeg_paths_dict[directory]:
            graph_path = directory + graph_id
            paths.append(graph_path)

    mp.freeze_support()

    if use_multiprocessing:
        with mp.Pool(n_jobs) as pool:
            results = pool.map(DegTime, paths)

    else:
        results = [DegTime(path) for path in paths]

    col_names = ['graph_group', 'graph_id', 'weighted_alg', 'n_nodes', 'n_edges', 'weighted', 'directed', 'time', 'path_centrality', 'path_length', 'path_nodes_traversed', 'path', 'diam_centrality', 'diam', 'diam_nodes_traversed', 'diam_path']

    df_soln = pd.DataFrame(results, columns=col_names)

    results_path = f'results/degree-unweighted/{datetime.now().strftime("%Y-%m-%d-%H%M%S")}.csv'
    df_soln.to_csv(results_path, index=False)

    # MVP degree timing test

    MVP_paths_dict = {
        './data/unweighted/undirected/barabasi_albert/100-barabasi_albert/': os.listdir('./data/unweighted/undirected/barabasi_albert/100-barabasi_albert/'),
        './data/unweighted/undirected/watts-strogatz/100-ws-10/': os.listdir('./data/unweighted/undirected/watts-strogatz/100-ws-10/'),
        './data/unweighted/undirected/watts-strogatz/100-ws-20/': os.listdir('./data/unweighted/undirected/watts-strogatz/100-ws-20/'),
        './data/unweighted/undirected/barabasi_albert/500-barabasi_albert/': os.listdir('./data/unweighted/undirected/barabasi_albert/500-barabasi_albert/'),
        './data/unweighted/undirected/watts-strogatz/500-ws-10/': os.listdir('./data/unweighted/undirected/watts-strogatz/500-ws-10/'),
        './data/unweighted/undirected/watts-strogatz/500-ws-20/': os.listdir('./data/unweighted/undirected/watts-strogatz/500-ws-20/'),
        './data/unweighted/undirected/barabasi_albert/1000-barabasi_albert/': os.listdir('./data/unweighted/undirected/barabasi_albert/1000-barabasi_albert/'),
        './data/unweighted/undirected/watts-strogatz/1000-ws-10/': os.listdir('./data/unweighted/undirected/watts-strogatz/1000-ws-10/'),
        './data/unweighted/undirected/watts-strogatz/1000-ws-20/': os.listdir('./data/unweighted/undirected/watts-strogatz/1000-ws-20/'),
        './data/unweighted/undirected/real-instances/': ['krebs_62.csv', 'dolphins_62.csv', 'sandi_auths_86.csv', 'USAir97_332_2126.csv', 'ieeebus_118.csv', 'santafe_118.csv', 'bus_662_906.csv', 'email_1133.csv', 'cerevisae_1458_1948.csv'],
    }

    paths = []

    for directory in MVP_paths_dict:
        for graph_id in MVP_paths_dict[directory]:
            graph_path = directory + graph_id
            paths.append(graph_path)

    mp.freeze_support()

    if use_multiprocessing:
        with mp.Pool(n_jobs) as pool:
            results = pool.map(MVPTime, paths)

    else:
        results = [MVPTime(path) for path in paths]

    col_names = ['graph_group', 'graph_id', 'n_nodes', 'n_edges', 'weighted', 'directed', 'time', 'path_centrality', 'path_length', 'path']

    df_soln = pd.DataFrame(results, columns=col_names)

    results_path = f'results/MVP/{datetime.now().strftime("%Y-%m-%d-%H%M%S")}.csv'
    df_soln.to_csv(results_path, index=False)

    ## Weighted degree timing test

    wdeg_paths_dict = {
        './data/weighted/directed/real-instances/': os.listdir('./data/unweighted/directed/real-instances/'),
        './data/weighted/undirected/real-instances/': os.listdir('./data/unweighted/undirected/real-instances/'),
    }

    paths = []

    for directory in wdeg_paths_dict:
        for graph_id in wdeg_paths_dict[directory]:
            graph_path = directory + graph_id
            paths.append(graph_path)

    mp.freeze_support()

    if use_multiprocessing:
        with mp.Pool(n_jobs) as pool:
            results = pool.map(DegWeightedTime, paths)

    else:
        results = [DegWeightedTime(path) for path in paths]

    col_names = ['graph_group', 'graph_id', 'weighted_alg', 'n_nodes', 'n_edges', 'weighted', 'directed', 'time', 'path_centrality', 'path_length', 'path_nodes_traversed', 'path', 'diam_centrality', 'diam', 'diam_nodes_traversed', 'diam_path']

    df_soln = pd.DataFrame(results, columns=col_names)

    results_path = f'results/degree-weighted/{datetime.now().strftime("%Y-%m-%d-%H%M%S")}.csv'
    df_soln.to_csv(results_path, index=False)

    ## Betweenness timing test

    bet_paths_dict = {
        './data/unweighted/undirected/barabasi_albert/100-barabasi_albert/': os.listdir('./data/unweighted/undirected/barabasi_albert/100-barabasi_albert/'),
        './data/unweighted/undirected/watts-strogatz/100-ws-10/': os.listdir('./data/unweighted/undirected/watts-strogatz/100-ws-10/'),
        './data/unweighted/undirected/watts-strogatz/100-ws-20/': os.listdir('./data/unweighted/undirected/watts-strogatz/100-ws-20/'),
        './data/unweighted/undirected/barabasi_albert/500-barabasi_albert/': os.listdir('./data/unweighted/undirected/barabasi_albert/500-barabasi_albert/'),
        './data/unweighted/undirected/watts-strogatz/500-ws-10/': os.listdir('./data/unweighted/undirected/watts-strogatz/500-ws-10/'),
        './data/unweighted/undirected/watts-strogatz/500-ws-20/': os.listdir('./data/unweighted/undirected/watts-strogatz/500-ws-20/'),
        './data/unweighted/undirected/barabasi_albert/1000-barabasi_albert/': os.listdir('./data/unweighted/undirected/barabasi_albert/1000-barabasi_albert/'),
        './data/unweighted/undirected/watts-strogatz/1000-ws-10/': os.listdir('./data/unweighted/undirected/watts-strogatz/1000-ws-10/'),
        './data/unweighted/undirected/watts-strogatz/1000-ws-20/': os.listdir('./data/unweighted/undirected/watts-strogatz/1000-ws-20/'),
        './data/weighted/directed/real-instances/': ['copenhagen-calls-directed.csv', 'copenhagen-sms-directed.csv'],
        './data/weighted/undirected/real-instances/': ['copenhagen-calls-undirected.csv', 'copenhagen-sms-undirected.csv'],
        './data/unweighted/directed/real-instances/': ['copenhagen-calls-directed.csv', 'copenhagen-sms-directed.csv'],
        './data/unweighted/undirected/real-instances/': ['krebs_62.csv', 'dolphins_62.csv', 'sandi_auths_86.csv', 'copenhagen-calls-undirected.csv', 'copenhagen-sms-undirected.csv',  'USAir97_332_2126.csv', 'ieeebus_118.csv', 'santafe_118.csv', 'bus_662_906.csv'],
    }

    paths = []

    for directory in bet_paths_dict:
        for graph_id in bet_paths_dict[directory]:
            graph_path = directory + graph_id
            paths.append(graph_path)

    mp.freeze_support()

    if use_multiprocessing:
        with mp.Pool(n_jobs) as pool:
            results = pool.map(BetweennessTime, paths)

    else:
        results = [BetweennessTime(path) for path in paths]

    col_names = ['graph_group', 'graph_id', 'n_nodes', 'n_edges', 'weighted', 'directed', 'time', 'path_length', 'path_centrality', 'path_nodes_traversed', 'path', 'diam', 'diam_centrality', 'diam_nodes_traversed', 'diam_path', 'pp_time', 'it_count']

    df_soln = pd.DataFrame(results, columns=col_names)

    results_path = f'results/betweenness/{datetime.now().strftime("%Y-%m-%d-%H%M%S")}.csv'
    df_soln.to_csv(results_path, index=False)
