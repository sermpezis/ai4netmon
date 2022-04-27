import pandas as pd
from Analysis.similarity import select_from_similarity as sfs

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

    df_ = data.copy()
    df_.index = df_.index.astype(str, copy=False)

    selected_from_greedy_least_similar = {}
    for time in range(TOTAL_TIMES):
        selected_from_greedy_least_similar[str(time)] = sfs.select_from_similarity_matrix(similarity_matrix=df_,
                                                                                      method='Greedy max',
                                                                                      nb_items=NB_ITEMS)
    pd.DataFrame(selected_from_greedy_least_similar).to_csv("selected_from_greedy_least_similar.csv")



select_from_kmeans10(df)
# select_from_kmeans20(df)
# select_from_greedy_leastsimilar(df)
