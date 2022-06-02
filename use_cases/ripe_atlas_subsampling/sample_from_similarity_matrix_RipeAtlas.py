import pandas as pd
from ai4netmon.Analysis.similarity import select_from_similarity as sfs
import numpy as np
import json

# set total times of runs for the selection methods

TOTAL_TIMES = 10

# set the path of the file with Ripe Atlas ASNs similarities

filename = './ripe_atlas_probe_asn_similarity_jaccard_paths_v4_median75_asn_max_20211124.csv'

df = pd.read_csv(filename, header=0, index_col=0)

df.values[[np.arange(df.shape[0])]*2] = 1
# methods to fill NaNs: with 0 or with the mean of column values
df = df.fillna(0)
# df = df.fillna(df.mean())

NB_ITEMS = 1000
SAVE_PATH = './data/'


def select_from_clustering(data, clusters, clustering_method, fill_nan_method='0'):
    """
    This function takes as input the dataframe of similarities and the number of clusters wanted, creates a dictionary that stores
    the ASNs collected each time the clustering method with selected clusters is performed, in the form of (k,v), where the key
    is the time the clustering algorithm runs, and value is a list of 1000 monitors that are collected.
    Then, the dictionary is stored to a csv in the form of dataframe.
    The function also saves the clusters themselves, with all the asns they contain.
    :param data: the input dataframe
    :param clusters: number of clusters for the algorithm to create
    :param clustering_method: choose between two clustering algorithms, Kmeans or SpectralClustering
    :param fill_nan_method: method to fill nans in the df
    """
    selected_from_clustering = {}

    for time in range(TOTAL_TIMES):
        selected_from_clustering[str(time)], _ = sfs.select_from_similarity_matrix(similarity_matrix=data, method='Clustering',
                                                                            clustering_method=clustering_method, nb_clusters=clusters,
                                                                            nb_items=NB_ITEMS)

    for key, value in selected_from_clustering.items():
        if len(value) < NB_ITEMS:
            while len(value) < NB_ITEMS:
                value.append('NaN')

    _, dict_of_clusters = sfs.select_from_similarity_matrix(similarity_matrix=data,
                                                                               method='Clustering',
                                                                               clustering_method=clustering_method,
                                                                               nb_clusters=clusters,
                                                                               nb_items=NB_ITEMS)

    for key, value in dict_of_clusters.items():
        if len(value) < max(len(item) for item in dict_of_clusters.values()):
            while len(value) < max(len(item) for item in dict_of_clusters.values()):
                value.append('NaN')


    print(dict_of_clusters)
    # json.dump(dict_of_clusters, open(SAVE_PATH+"clusters_lists_of_{}_{}_{}".format(clustering_method, clusters, fill_nan_method), 'w'))

    # pd.DataFrame(dict_of_clusters).to_csv(SAVE_PATH+"clusters_lists_of_{}_{}_{}.csv".format(clustering_method, clusters, fill_nan_method))
    
    pd.DataFrame(selected_from_clustering).to_csv(SAVE_PATH+"selected_from_{}_{}_{}.csv".format(clustering_method, clusters, fill_nan_method))


def select_from_greedy_leastsimilar(data):
    """
    This function takes as input the dataframe of similarities and first copies it,
    and changes the type of index to string, in order to be compatible with the greedy algorithm that is used.
    It creates a dictionary which stores the ASNs collected each time the greedy algorithm, that is based
    on the minimum aggregated similarity of items, is performed, in the form of (k,v), where the key is
    the time the algorithm runs, and value is a list of 1000 monitors that are collected.
    Then, the dictionary is stored to a csv in the form of dataframe.
    """

    # df_ = data.copy()
    data.index = data.index.astype(str, copy=False)

    selected_from_greedy_least_similar = {}
    for time in range(TOTAL_TIMES):
        selected_from_greedy_least_similar[str(time)] = sfs.select_from_similarity_matrix(similarity_matrix=data,
                                                                                      method='Greedy max',
                                                                                      nb_items=NB_ITEMS)
    pd.DataFrame(selected_from_greedy_least_similar).to_csv("selected_from_greedy_least_similar.csv")


def map_function(selected_probes, dict_probes, sampling_method):
    """
    Function that maps the sampled probes dictionary (which are the ids in the data_of_ripe_atlas_probs text) to
    their corresponding asns, for all ten sampled probes, and saves them to dataframe which is later written to csv
    file.
    Takes as input the dictionary of sampled probes, the dictionary of probes info loaded, and the sampling method
    (kmeans or greedy least)
    """
    list_of_asn_dicts = []

    for time in range(TOTAL_TIMES):
        asn_dicts = []

        for probe in selected_probes[str(time)]:
            if probe is not None:
                asn_dicts.append(list(filter(lambda dict_probes: dict_probes['id'] == int(probe), dict_probes)))

        list_of_asn_dicts.append(asn_dicts)

    final_asns = []

    for time in range(TOTAL_TIMES):
        asns_ = []

        for i in range(len(list_of_asn_dicts[time])):
            asns_.append([a_dict['asn_v4'] for a_dict in list_of_asn_dicts[time][i]])

        final_asns.append(asns_)

    for i in range(TOTAL_TIMES):
        final_asns[i] = [item for sublist in final_asns[i] for item in sublist]

    df_ = pd.DataFrame(final_asns)
    df_ = df_.transpose()
    df_.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(df_)

    pd.DataFrame(df_).to_csv("selected_from_{}_asns_of_probes.csv".format(sampling_method))




