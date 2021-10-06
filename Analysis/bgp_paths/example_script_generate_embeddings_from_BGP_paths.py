import networkx as nx
import bgp_paths_library as bpl


paths = [[1,2,3], [2,4,5], [5,3,1]]
method = 'example'
dict_args = {'embedding_size':4, 'default_value':22}
data = bpl.generate_ASN_embeddings_from_BGP_paths(paths, method, **dict_args)
print(data)