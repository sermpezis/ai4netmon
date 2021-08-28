import pandas as pd
import networkx as nx
import tempfile
import os.path
from node2vec import Node2Vec
from gensim.models import Word2Vec

# temp_dir = tempfile.TemporaryDirectory()
# print(temp_dir.name)

data = pd.read_csv('../Datasets/AS-relationships/20210701.as-rel2.txt', sep="|", skiprows=180)
data.columns = ['node1', 'node2', 'link', 'protocol']
data.drop(['protocol'], axis=1, inplace=True)
print(data)

graph = nx.Graph()
graph.add_nodes_from(data.node1, node_type="node")
graph.add_nodes_from(data.node2, node_type="node")

for line in data.values:
    # Due to weight = 0 Node2Vec does not work
    # graph.add_edge(line[0], line[1], weight=line[2])
    graph.add_edge(line[0], line[1])

print("=====================================")
print("Stats about the data")
print("=====================================")

print(nx.info(graph))

if not os.path.isfile('./embeddings.emb'):
    # FILES
    EMBEDDING_FILENAME = './embeddings.emb'
    EMBEDDING_MODEL_FILENAME = './embeddings.model'

    # Precompute probabilities and generate walks
    node2vec = Node2Vec(graph, dimensions=64, walk_length=10, num_walks=80, workers=1)
    # only for big graphs --> Slower
    # node2vec = Node2Vec(graph, dimensions=64, walk_length=10, num_walks=80, workers=1, temp_folder=temp_dir.name)
    model = node2vec.fit(window=5, min_count=1, batch_words=4)

    # Look for most similar nodes
    model.wv.most_similar('2')  # Output node names are always strings

    # Save embeddings for later use
    model.wv.save_word2vec_format(EMBEDDING_FILENAME)

    # Save model for later use
    model.save(EMBEDDING_MODEL_FILENAME)

    # temp_dir.cleanup()
else:
    # Load model after Node2Vec.save
    model = Word2Vec.load('./embeddings.model')
    print(model)

# Directed Graph
# G=nx.read_edgelist('Datasets/AS-relationships/20210701.as-rel2.txt',
#                         create_using = nx.DiGraph(), nodetype = None, data = [('weight', int)])#read graph
#
# model = Node2Vec(G, walk_length = 10, num_walks = 80,p = 0.25, q = 4, workers = 1)#init model
# model.train(window_size = 5, iter = 3)# train model
# embeddings = model.get_embeddings()# get embedding vectors
