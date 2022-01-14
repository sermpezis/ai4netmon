import pandas as pd
import numpy as np
import os
import json
import random
from sklearn.cluster import KMeans, SpectralClustering
from matplotlib import pyplot as plt
from tqdm import tqdm
from collections import defaultdict

MAX_LENGTH = 20
INITIAL_PROXIMITY = 1443060
ASN2ASN_DIST_FNAME = '../../Datasets/RIPE_RIS_peers/asn2asn__only_peers_pfx.json'
SIMILARITY_MATRIX = 'calculate_distance_and_similarity/ALL_RIPE_RIS_similarity_embeddings_BGP2VEC_20210107.csv'
PROXIMITY_FNAME = 'proximity_selected_monitors_ripe_ris_pathlens_100k.json'
ONLY_v4 = True
ONLY_v6 = False


def calculate_proximity(next_item, asn2asn, proximity):
    for o_asn, dict_o_asn in asn2asn.items():
        if (next_item in dict_o_asn.keys()) and (dict_o_asn[next_item] < proximity[o_asn]):
            proximity[o_asn] = dict_o_asn[next_item]
    return proximity


def get_proximity_vector(selected_items, asn2asn):
    proximity = {o_asn: MAX_LENGTH for o_asn in asn2asn.keys()}
    proximity_vector = []
    for i, item in tqdm(enumerate(selected_items)):
        proximity = calculate_proximity(item, asn2asn, proximity)
        proximity_vector.append(sum(proximity.values()))
    proximity_vector = [(i - 1) / INITIAL_PROXIMITY for i in proximity_vector]  # normalize proximity to [0,1]
    return proximity_vector


def get_argmax_total_similarity(similarity_matrix, from_items=None, rank_normalization=False):
    '''
    Finds the item of a matrix (similarity_matrix) that has the maximum aggregate similarity to all other items.
    If the "from_items" is not None, then only the rows/columns of the matrix in the from_items list are taken into account.
    :param  similarity_matrix:  (pandas.DataFrame) an NxN dataframe; should be (a) symmetric and (b) values {i,j} to
                                represent the similarity between item of row i and column j
    :param  from_items:         (list/set) a subset of the items (rows/columns) from which the item with the max similiarity will be selected
    :param  rank_normalization: (boolean) whether to modify the similarity matrix giving more emphasis to most similar values per row
                                by dividing each element with the rank it appears in the sorted list of values of the row
                                e.g., a_row = [0.5, 0.3, 0.4] --> modified_row = [0.5/1, 0.3/3, 0.4/2] = [0.5, 0.1, 0.2]
                                e.g., a_row = [0.1, 0.1, 0.4] --> modified_row = [0.1/2, 0.1/3, 0.4/1] = [0.05, 0.033, 0.4]
    :return:                    (scalar, e.g., str or int) the index of the item in the dataframe that has the max total similarity
    '''
    if from_items is None:
        df = similarity_matrix.copy()
    else:
        df = similarity_matrix.loc[from_items, from_items].copy()

    np.fill_diagonal(df.values, np.nan)  # set self-similarity to nan so that it is not taken into account

    if rank_normalization:
        for p1 in df.index:
            sorted_indexes = list(df.loc[p1, :].sort_values(ascending=False).index)
            df.loc[p1, sorted_indexes] = df.loc[p1, sorted_indexes] * [1.0 / i for i in range(1, 1 + df.shape[0])]

    sum_similarities = np.nansum(df, axis=1)
    if np.max(sum_similarities) == 0:  # all similarities are nan or zero
        next_item = random.sample(from_items, 1)[0]
    else:
        next_item = df.index[np.argmax(sum_similarities)]

    return next_item


def greedy_most_similar_elimination(similarity_matrix, rank_normalization=False):
    '''
    Selects iteratively the item in the given similarity_matrix that has the maximum aggregate similarity to all other items. At each iteration,
    only the similarities among the non-selected items are taken into account. At each iteration, the selected item is placed in the beginning of
    a list. At the end, this list is returned. Example: returned_list = [item_selected_last, ..., item_selected_first]
    :param  similarity_matrix:  (pandas.DataFrame) an NxN dataframe; should be (a) symmetric and (b) values {i,j} to
                                represent the similarity between item of row i and column j
    :param  rank_normalization: (boolean) whether to modify the similarity matrix giving more emphasis to most similar values per row
    :return:                    (list) a list of ordered items (from the input's index); the first item is the least similar
    '''
    selected_items = []
    for i in range(similarity_matrix.shape[0]):
        from_items = list(set(similarity_matrix.index) - set(selected_items))
        next_item = get_argmax_total_similarity(similarity_matrix, from_items=from_items,
                                                rank_normalization=rank_normalization)
        selected_items.insert(0, next_item)
    return selected_items


def get_argmin_total_similarity(similarity_matrix, from_items=None):
    '''
    Finds the item of a matrix (similarity_matrix) that has the minimum aggregate similarity to all other items.
    If the "from_items" is not None, then only the (a) rows of the matrix in the from_items list and (b) the columns
    of the matrix NOT in the from_items list are taken into account.
    :param  similarity_matrix:  (pandas.DataFrame) an NxN dataframe; should be (a) symmetric and (b) values {i,j} to
                                represent the similarity between item of row i and column j
    :param  from_items:         (list/set) a subset of the items (rows/columns) from which the item with the min similiarity will be selected
    :return:                    (scalar, e.g., str or int) the index of the item in the dataframe that has the min total similarity
    '''
    df = similarity_matrix.copy()
    np.fill_diagonal(df.values, np.nan)  # set self-similarity to nan so that it is not taken into account
    if from_items is not None:
        other_items = list(set(df.index) - set(from_items))
        df = df.loc[from_items, other_items]

    sum_similarities = np.nansum(df, axis=1)
    if np.max(sum_similarities) == 0:  # all similarities are nan or zero
        next_item = random.sample(from_items, 1)[0]
    else:
        next_item = df.index[np.argmin(sum_similarities)]

    return next_item


def greedy_least_similar_selection(similarity_matrix, nb_items=None):
    '''
    Selects iteratively the item in the given similarity_matrix that has the minimum aggregate similarity to all other items. At each iteration,
    only the similarities among the non-selected items and the already selected items are taken into account. At each iteration, the selected item is
    placed in the end of a list. At the end, this list is returned. Example: returned_list = [item_selected_first, ..., item_selected_last]
    :param  similarity_matrix:  (pandas.DataFrame) an NxN dataframe; should be (a) symmetric and (b) values {i,j} to
                                represent the similarity between item of row i and column j
    :param  nb_items:           (int) number of items to be selected; if None all items are selected in the returned list
    :return:                    (list) a list of ordered items (from the input's index); the first item is the least similar
    '''
    selected_items = []

    nb_total_items = similarity_matrix.shape[0]
    if (nb_items is None) or (nb_items > nb_total_items):
        nb_items = nb_total_items

    for i in range(nb_items):
        if len(selected_items) == 0:
            from_items = None
        else:
            from_items = list(set(similarity_matrix.index) - set(selected_items))
        next_item = get_argmin_total_similarity(similarity_matrix, from_items=from_items)
        selected_items.append(next_item)

    return selected_items


def random_selection(similarity_matrix, nb_items=None):
    """
    Selects randomly an item from the given similarity_matrix
    :param  similarity_matrix:  (pandas.DataFrame) an NxN dataframe; should be (a) symmetric and (b) values {i,j} to
                                represent the similarity between item of row i and column j
    :param  nb_items:           (int) number of items to be selected; if None all items are selected in the returned list
    :return:                    (list) a list of random items
    """

    selected_items = []

    nb_total_items = similarity_matrix.shape[0]
    if (nb_items is None) or (nb_items > nb_total_items):
        nb_items = nb_total_items

    for i in range(nb_items):
        temp = random.sample(list(similarity_matrix), 1)[0]
        selected_items.append(temp)

    return selected_items


def sample_from_clusters(cluster_members_dict, nb_items=None):
    '''
    Samples items from the clusters, starting from a random item in the largest cluster, then a random item in the second largest cluster, and so on.
    When elements of all clusters are selected, then starts again from the largest cluster, until all items (or up to nb_items) are selected.
    :param  cluster_members_dict:   (dict of lists) dict of the form {cluster label: list of members of the cluster}
    :param  nb_items:               (int) number of items to be selected; if None all items are selected in the returned list
    :return:                        (list) a list of ordered items that are the samples from clusters
    '''

    nb_clusters = len(cluster_members_dict.keys())
    nb_all_items = sum([len(v) for v in cluster_members_dict.values()])
    if (nb_items is None) or (nb_items > nb_all_items):
        nb_items = nb_all_items

    sorted_clusters = sorted(cluster_members_dict, key=lambda k: len(cluster_members_dict.get(k)), reverse=True)

    selected_items = []
    for i in range(nb_items):
        ind = i % nb_clusters  # iterate over the sorted_clusters by getting the index of the current cluster
        current_cluster = sorted_clusters[ind]
        len_current_cluster = len(cluster_members_dict[current_cluster])
        if len_current_cluster > 0:
            next_item_ind = random.sample(range(len_current_cluster), 1)[0]
            next_item = cluster_members_dict[current_cluster].pop(next_item_ind)
            selected_items.append(next_item)
        i += 1

    return selected_items


def clustering_based_selection(similarity_matrix, clustering_method, nb_clusters, nb_items=None, **kwargs):
    '''
    Applies a clustering algorithm to the similarity matrix to cluster items, and then selects samples from the classes.
    :param  similarity_matrix:  (pandas.DataFrame) an NxN dataframe; should be (a) symmetric and (b) values {i,j} to
                                represent the similarity between item of row i and column j
    :param  nb_clusters:        (int) number of clusters
    :return:                    (list) a list of ordered items that are the samples from clusters
    '''
    sim = similarity_matrix.to_numpy()
    sim = np.nan_to_num(sim, nan=0)
    if clustering_method == 'SpectralClustering':
        clustering = SpectralClustering(n_clusters=nb_clusters, affinity='precomputed', **kwargs).fit(sim)
    elif clustering_method == 'Kmeans':
        clustering = KMeans(n_clusters=nb_clusters, **kwargs).fit(sim)
    else:
        raise ValueError
    cluster_members_dict = defaultdict(list)
    for i, label in enumerate(clustering.labels_):
        cluster_members_dict[label].append(similarity_matrix.index[i])

    return sample_from_clusters(cluster_members_dict, nb_items=nb_items)


def select_from_similarity_matrix(similarity_matrix, method, **kwargs):
    if method == 'Greedy min':
        selected_items = greedy_most_similar_elimination(similarity_matrix, **kwargs)
    elif method == 'Greedy max':
        selected_items = greedy_least_similar_selection(similarity_matrix, **kwargs)
    elif method == 'Clustering':
        selected_items = clustering_based_selection(similarity_matrix, **kwargs)
    elif method == 'Random':
        selected_items = random_selection(similarity_matrix, **kwargs)
    else:
        raise ValueError
    return selected_items


def plot_proximity_score_for_different_values_of_k(proximity_vector):
    fontsize = 15
    linewidth = 2
    colors = ['g', '--g', 'r', '--r', 'b', '--b', 'k', '--k', 'm', '--m']
    leg_str = []
    for i, k in enumerate(proximity_vector.keys()):
        X = list(range(1, 1 + len(proximity_vector[k])))
        plt.plot(X, proximity_vector[k], colors[i], linewidth=linewidth)
        leg_str.append(k)
    plt.xscale('log')
    plt.xlabel('#monitors', fontsize=fontsize)
    plt.ylabel('Proximity (normalized)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.axis([1, 1500, 0.1, 0.35])
    plt.legend(leg_str)
    plt.grid(True)
    # plt.savefig('fig_ripe_ris_subset_selection_vs_proximity.png')
    plt.show()


with open(ASN2ASN_DIST_FNAME, 'r') as f:
    asn2asn = json.load(f)

similarity_matrix = pd.read_csv(SIMILARITY_MATRIX, header=0, index_col=0)
similarity_matrix.replace(np.nan, 0, inplace=True)

# find full feeding peers
feed = defaultdict(lambda: 0)
for o_asn, dict_o_asn in asn2asn.items():
    for m_asn, dist in dict_o_asn.items():
        feed[m_asn] += 1
full_feeders = [m_asn for m_asn, nb_feeds in feed.items() if nb_feeds > 65000]

if ONLY_v4:
    peers_v4 = [m for m in similarity_matrix.index if ':' not in m]
    similarity_matrix = similarity_matrix.loc[peers_v4, peers_v4]
    full_feeders = set(full_feeders).intersection(set(peers_v4))
elif ONLY_v6:
    peers_v6 = [m for m in similarity_matrix.index if ':' in m]
    similarity_matrix = similarity_matrix.loc[peers_v6, peers_v6]
    full_feeders = set(full_feeders).intersection(set(peers_v6))

method_param_dict = {
    'Random ': {'method': 'Random', 'sim_matrix': similarity_matrix, 'args': {}},
    'Greedy min Geo': {'method': 'Greedy min', 'sim_matrix': similarity_matrix, 'args': {}},
    'Greedy min full Geo': {'method': 'Greedy min', 'sim_matrix': similarity_matrix.loc[full_feeders, full_feeders], 'args': {}},
    'Greedy max Geo': {'method': 'Greedy max', 'sim_matrix': similarity_matrix, 'args': {}},
    'Greedy max full Geo': {'method': 'Greedy max', 'sim_matrix': similarity_matrix.loc[full_feeders, full_feeders], 'args': {}},
    'Clustering spectral k7': {'method': 'Clustering', 'sim_matrix': similarity_matrix, 'args': {'clustering_method': 'SpectralClustering', 'nb_clusters': 7}},
    'Clustering spectral k7 full': {'method': 'Clustering', 'sim_matrix': similarity_matrix.loc[full_feeders, full_feeders], 'args': {'clustering_method': 'SpectralClustering', 'nb_clusters': 7}},
    'Clustering kmeans k7': {'method': 'Clustering', 'sim_matrix': similarity_matrix, 'args': {'clustering_method': 'Kmeans', 'nb_clusters': 7}},
    'Clustering kmeans k7 full': {'method': 'Clustering', 'sim_matrix': similarity_matrix.loc[full_feeders, full_feeders], 'args': {'clustering_method': 'Kmeans', 'nb_clusters': 7}}}

print('### Selected monitors by method ###')
for m, params in method_param_dict.items():
    selected_items = select_from_similarity_matrix(params['sim_matrix'], params['method'], **params['args'])
    print('\t{} [DONE]'.format(m))
    with open('dataset_selected_monitors_ripe_ris_pathlens_100k_{}.json'.format(
            '_'.join(m.lower().translate('()').split(' '))), 'w') as f:
        json.dump(selected_items, f)

# calculating the proximity vector takes ~20sec per method
if os.path.exists(PROXIMITY_FNAME):
    print('Loading proximities from existing file')
    with open(PROXIMITY_FNAME, 'r') as f:
        proximity_vector = json.load(f)
else:
    print('Calculating proximities')
    proximity_vector = dict()
    for m, params in method_param_dict.items():
        with open('dataset_selected_monitors_ripe_ris_pathlens_100k_{}.json'.format('_'.join(m.lower().split(' '))),
                  'r') as f:
            selected_items = json.load(f)
        proximity_vector[m] = get_proximity_vector(selected_items, asn2asn)
        print('\t{} [DONE]'.format(m))
    with open(PROXIMITY_FNAME, 'w') as f:
        json.dump(proximity_vector, f)

plot_proximity_score_for_different_values_of_k(proximity_vector)
