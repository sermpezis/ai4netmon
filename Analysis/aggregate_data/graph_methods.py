import networkx as nx
import urllib
import contextlib

def create_graph_from_AS_relationships(filename):
    if 'raw.githubusercontent.com' in filename: # if the given filename is an online resource 
        with contextlib.closing(urllib.request.urlopen(filename)) as f:
            G = nx.read_edgelist(f, comments='#', delimiter='|', nodetype=int, data=(('rel',int),('source',str)))
    else:
        G = nx.read_edgelist(filename, comments='#', delimiter='|', nodetype=int, data=(('rel',int),('source',str)))
    return G

def get_stubs_from_AS_graph(G):
    return [n[0] for n in G.degree() if n[1]<=1]