import networkx as nx
import pandas as pd
import bgp_paths_library as bpl
from data_generator import generate_random_numbers
from Analysis.aggregate_data.data_aggregation_tools import create_graph_from_bgp_paths

method = 'bgp2vec_paths'

if method == 'example':
    paths = [[1, 2, 3], [2, 4, 5], [5, 3, 1]]
    dict_args = {'embedding_size': 4, 'default_value': 22}
    data = bpl.generate_ASN_embeddings_from_BGP_paths(paths, method, **dict_args)
    print(data)
elif method == 'bgp2vec_paths':
    paths = generate_random_numbers()
    bgp_graph = create_graph_from_bgp_paths(paths)
    print(print(nx.info(bgp_graph)))
    dict_args = {'dimensions': 64, 'walk_length': 100, 'num_walks': 5, 'workers': 1, 'window': 5, 'min_count': 2, 'batch_words': 4}
    data = bpl.generate_ASN_embeddings_from_BGP_paths(paths, method, bgp_graph, **dict_args)
    print(data)
else:
    raise Exception('Not defined type of embeddings')