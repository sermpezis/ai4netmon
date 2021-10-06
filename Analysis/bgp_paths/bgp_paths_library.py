import pandas as pd

def my_example_embeddings_method(paths, embedding_size, default_value=1):
    """
    :param paths: (list) a list of BGP paths; a BGP path is a list of integers (ASNs)
    :param embedding_size: (int) the size of the embedding
    :param default_value: (int) the value for the embeddings
    :return: (pandas dataframe object) a dataframe with index the ASN numbers included in the paths where each row has <embedding_size> embeddings all with the same <default_value>
    """
    unique_ASNs = set()
    for path in paths:
        unique_ASNs.update(path)
    columns = ['embedding_'+str(i) for i in range(embedding_size)]
    data = pd.DataFrame(default_value, index=unique_ASNs, columns=columns)
    return data

# TODO: @Christos: if your are not familiar with **kwargs, check some online tutorial (e.g., https://www.geeksforgeeks.org/args-kwargs-python/, https://realpython.com/python-kwargs-and-args/)
def generate_ASN_embeddings_from_BGP_paths(paths, method, **kwargs):
    """
    :param paths: (list) a list of BGP paths; a BGP path is a list of integers (ASNs)
    :param method: (string) the method to be used to generate the embeddings
    :param kwargs: (dictionary) a dictionary of arguments to be passed to the method that will generate the embeddings; e.g., kwargs={'embedding_size':10, 'alpha_param':1}
    :return: (pandas dataframe object) a dataframe with index the ASN numbers included in the paths
    """
    if method == 'example':
        data = my_example_embeddings_method(paths, **kwargs)
    elif method == 'word2vec':
        # TODO
        pass
    else:
        raise Exception('Not defined method for generation of graph embeddings')
    return data

