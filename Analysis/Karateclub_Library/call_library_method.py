import networkx as nx
import pandas as pd
import numpy as np
import os.path
from karateclub import Node2Vec

if not os.path.isfile('./convert_to_consecutive.csv'):

    PATH_AS_RELATIONSHIPS = '../../Datasets/AS-relationships/20210701.as-rel2.txt'
    data = pd.read_csv(PATH_AS_RELATIONSHIPS, sep="|", skiprows=180, header=None)
    data.columns = ['source', 'target', 'link', 'protocol']
    data.drop(['link', 'protocol'], axis=1, inplace=True)

    unique_nodes1 = set(data.source)
    unique_nodes2 = set(data.target)

    all_nodes = set(unique_nodes1.union(unique_nodes2))
    sort_nodes = sorted(all_nodes)

    # create consecutive numbers
    temp_list = list(range(len(sort_nodes)))

    # Create dict: Key is the real node and key is the consecutive node
    dict = {k: v for k, v in zip(sort_nodes, temp_list)}

    for key, value in dict.items():
            data['source'][data['source'] == key] = value
            data['target'][data['target'] == key] = value
    data.to_csv("convert_to_consecutive.csv", sep=',', index=False)
else:
    data = pd.read_csv('convert_to_consecutive.csv', sep=",")


graph = nx.Graph()
graph.add_nodes_from(data.source, node_type="node")
graph.add_nodes_from(data.target, node_type="node")

for line in data.values:
    graph.add_edge(line[0], line[1])

dimensions = 64
model = Node2Vec(dimensions=dimensions, walk_length=5, p=0.5, q=2.0, window_size=2, epochs=3)
model.fit(graph)
embedding = model.get_embedding()

id = [i for i in range(0, len(embedding))]
embedding = np.insert(embedding, 0, id, axis=1)
pd.DataFrame(embedding).to_csv("Node2Vec_64_wl5_ws2_ep3.csv", sep=',', index=False)

