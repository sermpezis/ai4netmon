import networkx as nx
import graph_embeddings_library as gel


graph = nx.Graph()
graph.add_nodes_from(['a','b','c'])
method = 'example'
dict_args = {'embedding_size':4, 'default_value':22}
data = gel.generate_ASN_embeddings_from_graph(graph, method, **dict_args)
print(data)