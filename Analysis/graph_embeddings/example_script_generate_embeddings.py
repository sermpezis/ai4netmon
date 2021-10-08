import networkx as nx
import graph_embeddings_library as gel
from Analysis.aggregate_data.data_aggregation_tools import create_graph_from_AS_relationships

graph = nx.Graph()
method = 'node2vec'

if method == 'example':
    graph = nx.Graph()
    graph.add_nodes_from(['a', 'b', 'c'])
    method = 'example'
    dict_args = {'embedding_size': 4, 'default_value': 22}
    data = gel.generate_ASN_embeddings_from_graph(graph, method, **dict_args)
elif method == 'node2vec':
    graph = create_graph_from_AS_relationships()
    print(nx.info(graph))
    dict_args = {'dimensions': 64, 'walk_length': 10, 'num_walks': 80, 'workers': 1, 'window': 5, 'min_count': 1, 'batch_words': 4}
    data = gel.generate_ASN_embeddings_from_graph(graph, method, **dict_args)
else:
    raise Exception('Not defined type of embeddings')