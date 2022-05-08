import pandas as pd
from ai4netmon.Analysis.similarity import select_from_similarity as sfs
import numpy as np
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


def select_from_kmeans10(data):
    """
    This function takes as input the dataframe of similarities, creates a dictionary that stores
    the ASNs collected each time the k-means with 10 clusters is performed, in the form of (k,v), where the key
    is the time the k-means algorithm runs, and value is a list of 1000 monitors that are collected.
    Then, the dictionary is stored to a csv in the form of dataframe.
    """
    selected_from_k_means_10 = {}
    for time in range(TOTAL_TIMES):
        selected_from_k_means_10[str(time)] = sfs.select_from_similarity_matrix(similarity_matrix=data, method='Clustering',
                                                                            clustering_method='Kmeans', nb_clusters=10,
                                                                            nb_items=NB_ITEMS)

    pd.DataFrame(selected_from_k_means_10).to_csv("selected_from_k_means_10.csv")


def select_from_kmeans20(data):
    """
    This function takes as input the dataframe of similarities, creates a dictionary that stores
    the ASNs collected each time the k-means with 20 clusters is performed, in the form of (k,v), where the key
    is the time the k-means algorithm runs, and value is a list of 500 monitors that are collected.
    Then, the dictionary is stored to a csv in the form of dataframe.
    """
    selected_from_k_means_20 = {}
    for time in range(TOTAL_TIMES):
        selected_from_k_means_20[str(time)] = sfs.select_from_similarity_matrix(similarity_matrix=data, method='Clustering',
                                                                            clustering_method='Kmeans', nb_clusters=20,
                                                                            nb_items=NB_ITEMS)
    for key, value in selected_from_k_means_20.items():
        if len(value) < NB_ITEMS:
            while len(value) < NB_ITEMS:
                value.append('NaN')

    pd.DataFrame(selected_from_k_means_20).to_csv("selected_from_k_means_20.csv")


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



