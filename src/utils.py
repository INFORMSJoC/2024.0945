import networkx as nx
import numpy as np
import pandas as pd

def read_saved_instance(path, directed=False, return_nxG=False, w_max=None, seed=None, clean_labels=False, to_string=False):
    '''
    Function to read csv file of saved graphs

    Inputs:
    path: str
        - Path to graph
    directed: bool
        - True if graph is directed, False if otherwise
    return_nxG: bool
        - True if networkx graph is to be returned as well, False if otherwise
    w_max: int; default:None
        - maximum weight to generate if random weights are desired
    seed: int; default:None
        - random seed if random weights are desired
    clean_labels: bool; default: False
        - True if nodes should be reindexed; False if otherwise
    to_string: bool; default: False
        - True nodes should be converted to strings; False if otherwise
    
    
    Returns:
    arcs: dict of dicts
        dict to dicts of arcs, with values set to weights
    G: networkx.DiGraph
        networkx directed graph, only returned if `return_nxG == True`
    '''

    df_edges = pd.read_csv(path, header=None)

    if to_string:
        df_edges[0] = df_edges[0].astype(str)
        df_edges[1] = df_edges[1].astype(str)

    # Unweighted graph
    if df_edges.shape[1] == 2:
        G = nx.from_pandas_edgelist(df_edges, 0, 1, create_using=nx.DiGraph())
        np.random.seed(seed)
        if w_max == None:
            w = 1
        for (u, v) in G.edges():
            if w_max != None:
                w = np.random.randint(1, w_max + 1)
            G.edges[u, v]['weight'] = w

    # Weighted graph
    elif df_edges.shape[1] == 3:
        df_edges.columns = [0, 1, 'weight']
        G = nx.from_pandas_edgelist(df_edges, 0, 1, 'weight', create_using=nx.DiGraph())
        if w_max != None:
            print('Ignoring `w_max` as graph has predefined weights.')

    else:
        raise TypeError(f'{path} does not have 2 or 3 columns.')
    
    if clean_labels:
        G = nx.convert_node_labels_to_integers(G)
    
    # Generating undirected graph if directed
    if not directed:
        G = G.to_undirected()

    arcs = nx.to_dict_of_dicts(G)
    arcs = {i: {j: arcs[i][j]['weight'] for j in arcs[i] if i != j} for i in arcs}

    if return_nxG:
        return arcs, G
    return arcs

def aug_weighted_to_unweighted(path, directed, return_nxG):
    '''
    Function to augment a weighted graph to become an unweighted graph by converting 
    edges with weight $w_{ij}$ to $w_{ij}$ new edges

    Inputs:
    path: str
        - Path to graph
    directed: bool
        - True if graph is directed, False if otherwise
    return_nxG: bool
        - True if networkx graph is to be returned as well, False if otherwise
    
    Returns:
    arcs: dict of dicts
        dict to dicts of arcs, with values set to weights
    G: networkx.DiGraph
        networkx directed graph, only returned if `return_nxG == True`
    '''

    df_edges = pd.read_csv(path, header=None)
    df_edges.columns = ['u', 'v', 'w']
    new_edges = []

    for index, row in df_edges.iterrows():
        u = str(row['u'])
        v = str(row['v'])
        w = row['w']
        if row['w'] == 1:
            new_edges.append([u, v, 1])
        else:
            new_edges.append([u, f'{u}_{v}_0', 1])
            for i in range(w - 2):                
                new_edges.append([f'{u}_{v}_{i}', f'{u}_{v}_{i+1}', 1])
            new_edges.append([f'{u}_{v}_{w - 2}', v, 1])

    new_df = pd.DataFrame(new_edges, columns=[0, 1, 'weight'])
    G = nx.from_pandas_edgelist(new_df, 0, 1, 'weight', create_using=nx.DiGraph())

    if not directed:
        G = G.to_undirected()

    arcs = nx.to_dict_of_dicts(G)
    arcs = {i: {j: arcs[i][j]['weight'] for j in arcs[i] if i != j} for i in arcs}

    if return_nxG:
        return arcs, G
    return arcs