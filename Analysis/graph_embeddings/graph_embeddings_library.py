import pandas as pd

GRAPH_EMBEDDING_METHODS = ['example','node2vec']

def my_example_embeddings_method(graph, embedding_size, default_value=1):
    """
    :param graph: (networkx Graph object) a graph, based on which the method generates embeddings for its nodes
    :param embedding_size: (int) the size of the embedding
    :param default_value: (int) the value for the embeddings
    :return: (pandas dataframe object) a dataframe where each node of the <graph> has <embedding_size> embeddings all with the same <default_value>
    """
    index = graph.nodes()
    columns = ['embedding_'+str(i) for i in range(embedding_size)]
    data = pd.DataFrame(default_value, index=index, columns=columns)
    return data

# TODO: @Christos: if your are not familiar with **kwargs, check some online tutorial (e.g., https://www.geeksforgeeks.org/args-kwargs-python/, https://realpython.com/python-kwargs-and-args/)
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
        # TODO: sth like the following: data = node2vec(graph, **kwargs)
        # TODO: when you populate the code here, delete the following line with the "pass"
        pass
    else:
        raise Exception('Not defined method for generation of graph embeddings')
    return data

