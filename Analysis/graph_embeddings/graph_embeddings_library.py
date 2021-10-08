import pandas as pd
import os.path
from node2vec import Node2Vec

GRAPH_EMBEDDING_METHODS = ['example', 'node2vec']


def my_example_embeddings_method(graph, embedding_size, default_value=1):
    """
    :param graph: (networkx Graph object) a graph, based on which the method generates embeddings for its nodes
    :param embedding_size: (int) the size of the embedding
    :param default_value: (int) the value for the embeddings
    :return: (pandas dataframe object) a dataframe where each node of the <graph> has <embedding_size> embeddings all with the same <default_value>
    """
    index = graph.nodes()
    columns = ['embedding_' + str(i) for i in range(embedding_size)]
    data = pd.DataFrame(default_value, index=index, columns=columns)
    return data


def node2vec(graph, dimensions, walk_length, num_walks, workers, window, min_count, batch_words):
    """
    :param graph: (networkx Graph object) a graph, based on which the method generates embeddings for its nodes
    :param dimensions: (int) Embedding dimensions
    :param walk_length: (int) Number of nodes in each walk
    :param num_walks: (int) Number of walks per node
    :param workers: (int) Number of workers for parallel execution
    :param window: (int) This parameter should be =1 when we run our program in WINDOWS
    :param min_count: (int)
    :param batch_words: (int)
    :return: A dataframe containing the embeddings that have been trained based on ASns
    """
    if not os.path.isfile('./embeddings.emb'):
        EMBEDDING_FILENAME = './embeddings.emb'

        node2vec_model = Node2Vec(graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
        model = node2vec_model.fit(window=window, min_count=min_count, batch_words=batch_words)

        # Look for most similar nodes. Output node names are always strings
        model.wv.most_similar('2')
        model.wv.save_word2vec_format(EMBEDDING_FILENAME)
        data = pd.read_table("embeddings.emb", header=None, sep=" ", skiprows=1)
    else:
        # Load model after Node2Vec.save
        data = pd.read_table("embeddings.emb", header=None, sep=" ", skiprows=1)
    return data


def generate_ASN_embeddings_from_graph(graph, method, **kwargs):
    """
    :param graph: (networkx Graph object) a graph, based on which the method generates embeddings for its nodes
    :param method: (string) the method to be used to generate graph embeddings
    :param kwargs: (dictionary) a dictionary of arguments to be passed to the method that will generate the embeddings; e.g., kwargs={'embedding_size':10, 'alpha_param':1}
    :return: (pandas dataframe object) a dataframe with index the ASN number of nodes and columns the graph embeddings for each node
    """
    if method == 'example':
        data = my_example_embeddings_method(graph, **kwargs)
    elif method == 'node2vec':
        data = node2vec(graph, **kwargs)
    else:
        raise Exception('Not defined method for generation of graph embeddings')
    return data