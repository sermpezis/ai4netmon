import networkx as nx

def create_graph_from_AS_relationships(filename):
    G = nx.read_edgelist(filename, comments='#', delimiter='|', nodetype=int, data=(('rel',int),('source',str)))
    return G

def get_stubs_from_AS_graph(G):
    return [n[0] for n in G.degree() if n[1]<=1]