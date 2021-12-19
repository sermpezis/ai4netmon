import pandas as pd
import numpy as np
import random
from collections import defaultdict

import json
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


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
        df = similarity_matrix.loc[from_items,from_items].copy()

    np.fill_diagonal(df.values, np.nan) # set self-similarity to nan so that it is not taken into account

    if rank_normalization:
        for p1 in df.index:
            sorted_indexes = list(df.loc[p1,:].sort_values(ascending=False).index)
            df.loc[p1,sorted_indexes] = df.loc[p1,sorted_indexes] * [1.0/i for i in range(1,1+df.shape[0])] 
    sum_similarities = np.nansum(df, axis=1)
    if np.max(sum_similarities) == 0: # all similarities are nan or zero
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
        from_items = list(set(similarity_matrix.index)-set(selected_items))
        next_item = get_argmax_total_similarity(similarity_matrix, from_items=from_items, rank_normalization=rank_normalization)
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
    np.fill_diagonal(df.values, np.nan) # set self-similarity to nan so that it is not taken into account
    if from_items is not None:
        other_items = list(set(df.index) - set(from_items))
        df = similarity_matrix.loc[from_items, other_items].copy()
    
    sum_similarities = np.nansum(df, axis=1)
    if np.max(sum_similarities) == 0: # all similarities are nan or zero
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
        if len(selected_items)==0:
            from_items = None
        else:
            from_items = list(set(similarity_matrix.index)-set(selected_items))
        next_item = get_argmin_total_similarity(similarity_matrix, from_items=from_items)
        selected_items.append(next_item)

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
        ind = i%nb_clusters # iterate over the sorted_clusters by getting the index of the current cluster 
        current_cluster = sorted_clusters[ind]
        len_current_cluster = len(cluster_members_dict[current_cluster])
        if len_current_cluster > 0:
            next_item_ind = random.sample(range(len_current_cluster),1)[0]
            next_item = cluster_members_dict[current_cluster].pop(next_item_ind)
            selected_items.append(next_item)
        i += 1

    return selected_items


def clustering_based_selection(similarity_matrix, clustering_method, nb_clusters, nb_items=None):
    '''
    Applies a clustering algorithm to the similarity matrix to cluster items, and then selects samples from the classes.
    :param  similarity_matrix:  (pandas.DataFrame) an NxN dataframe; should be (a) symmetric and (b) values {i,j} to 
                                represent the similarity between item of row i and column j
    :param  clustering_method:  (str) 'SpectralClustering' or 'Kmeans'    
    :param  nb_clusters:        (int) number of clusters
    :param  nb_items:           (int) number of items to be selected; if None all items are selected in the returned list                            
    :return:                    (list) a list of ordered items that are the samples from clusters
    '''
    if clustering_method == 'SpectralClustering':
        clustering = SpectralClustering(n_clusters=nb_clusters, affinity='precomputed').fit(similarity_matrix.to_numpy())
    elif clustering_method == 'Kmeans':
        pass
    else:
        raise ValueError

    cluster_members_dict = defaultdict(list)
    for i, label in enumerate(clustering.labels_):
        cluster_members_dict[label].append(similarity_matrix.index[i])
    print(cluster_members_dict)

    return sample_from_clusters(cluster_members_dict, nb_items=nb_items)




items = ['a','b','c','d']
similarity_matrix = pd.DataFrame([[1.0,0.5,0.9,0.2],[0.5,1.0,0.3,0.5],[0.9,0.3,1.0,0.7],[0.2,0.5,0.7,1.0]], columns=items, index=items)
print(similarity_matrix)
greedy_min = greedy_most_similar_elimination(similarity_matrix)
greedy_max = greedy_least_similar_selection(similarity_matrix)
greedy_min_mod = greedy_most_similar_elimination(similarity_matrix, rank_normalization=True)
cluster_members_dict = {'c1':['a','b'], 'c2':['c'], 'c3':['d']}
cluster_samples = clustering_based_selection(similarity_matrix, 'SpectralClustering', 1)
print(greedy_min)
print(greedy_max)
print(greedy_min_mod)
print(cluster_samples)
print(similarity_matrix)
exit()




SIMILARITY_MATRIX_FNAME = './data/similarity_matrix_distances_pathlens_100k.csv'
ASN2ASN_DIST_FNAME = './data/asn2asn__only_peers_pfx.json'
MAX_LENGTH = 20
NB_CLUSTERS = list(range(1,10))+[10, 20, 50, 100, 200]
INITIAL_PROXIMITY = 1443060#MAX_LENGTH*len(asn2asn.keys())



proximity = {o_asn:MAX_LENGTH for o_asn in asn2asn.keys()}
# proximity_min = [sum(proximity.values())]
proximity_min = []
for i,e in tqdm(enumerate(selected_elements_min)):
    if i==0:
        proximity = calculate_proximity(e, [], asn2asn, proximity)
    else:
        proximity = calculate_proximity(e, selected_elements_min[0:i-1], asn2asn, proximity)
    proximity_min.append(sum(proximity.values()))
proximity_min = [(i-1)/INITIAL_PROXIMITY for i in proximity_min]

















def calculate_proximity(new_monitor, existing_monitors, asn2asn, proximity):
    for o_asn, dict_o_asn in asn2asn.items():
        if (new_monitor in dict_o_asn.keys()) and (dict_o_asn[new_monitor] < proximity[o_asn]):
            proximity[o_asn] = dict_o_asn[new_monitor]
    return proximity

def calculate_proximity_set(monitors, asn2asn, proximity):
    for o_asn, dict_o_asn in asn2asn.items():
        monitors_with_path = [m for m in monitors if m in dict_o_asn.keys()]
        if (len(monitors_with_path) >0):
            proximity[o_asn] = min([dict_o_asn[m] for m in monitors_with_path]) 
    return proximity



## Method: greedy find max improvement
class OptEndException(Exception):
    pass

def find_greedy_opt(dist_matrix, removed_elements, asn2asn, proximity):
    remaining_elements = list(set(dist_matrix.columns) - set(removed_elements))
    assert len(remaining_elements) > 0, 'No more elements to be selected'        
    df = dist_matrix.loc[remaining_elements,remaining_elements]
    improvements = defaultdict(lambda : 0)
    for o_asn, dict_o_asn in asn2asn.items():
        min_distance = min(dict_o_asn.values())
        if (min_distance<proximity[o_asn]):
            min_dist_peers = [p for p,d in dict_o_asn.items() if d==min_distance]
            for p in min_dist_peers:
                improvements[p] += proximity[o_asn]-min_distance
    if len(improvements.keys()) == 0:
        raise OptEndException
    new_element = sorted(improvements, key=improvements.get)[-1]
    return new_element



# load similarity matrix
print('Loading distance matrix')
with open(SIMILARITY_MATRIX_FNAME, 'r') as f:
  distance_matrix = pd.read_csv(f,header=0)
distance_matrix.index = distance_matrix.columns

# load asn2asn dataset (to calculate improvements)
print('Loading asn2asn dataset')
with open(ASN2ASN_DIST_FNAME, 'r') as f:
  asn2asn = json.load(f)





# print('Greedy - opt')
# selected_elements_opt = []
# proximity = {o_asn:MAX_LENGTH for o_asn in asn2asn.keys()}
# proximity_opt = []
# for p in tqdm(distance_matrix.columns):
#     try:
#         next_monitor = find_greedy_opt(distance_matrix, selected_elements_opt, asn2asn, proximity)
#         if len(selected_elements_opt) == 0:
#             proximity = calculate_proximity(next_monitor, [], asn2asn, proximity)
#         else:
#             proximity = calculate_proximity(next_monitor, selected_elements_opt, asn2asn, proximity)
#         selected_elements_opt.append(next_monitor)
#         proximity_opt.append(sum(proximity.values()))
#     except OptEndException:
#         min_proximity = proximity_opt[-1]
#         proximity_opt.extend([min_proximity]*(len(distance_matrix.columns) - len(proximity_opt)))
#         break
# proximity_opt = [(i-1)/INITIAL_PROXIMITY for i in proximity_opt]


# print('Greedy max')
# selected_elements_max = []
# proximity = {o_asn:MAX_LENGTH for o_asn in asn2asn.keys()}
# # proximity_max = [sum(proximity.values())]
# proximity_max = []
# for i in tqdm(range(len(distance_matrix.columns))):
#     next_monitor = find_max_distance(distance_matrix, selected_elements_max)
#     proximity = calculate_proximity(next_monitor, selected_elements_max, asn2asn, proximity)
#     selected_elements_max.append(next_monitor)
#     proximity_max.append(sum(proximity.values()))
# proximity_max = [(i-1)/INITIAL_PROXIMITY for i in proximity_max]


# print('Greedy min')
# selected_elements_min = []
# remaining_elements = set(distance_matrix.columns)
# for i in tqdm(range(len(distance_matrix.columns))):
#     next_monitor = find_min_distance(distance_matrix, list(set(distance_matrix.columns)-set(remaining_elements)))
#     remaining_elements.remove(next_monitor)
#     selected_elements_min.insert(0,next_monitor)

# proximity = {o_asn:MAX_LENGTH for o_asn in asn2asn.keys()}
# # proximity_min = [sum(proximity.values())]
# proximity_min = []
# for i,e in tqdm(enumerate(selected_elements_min)):
#     if i==0:
#         proximity = calculate_proximity(e, [], asn2asn, proximity)
#     else:
#         proximity = calculate_proximity(e, selected_elements_min[0:i-1], asn2asn, proximity)
#     proximity_min.append(sum(proximity.values()))
# proximity_min = [(i-1)/INITIAL_PROXIMITY for i in proximity_min]

# print('Greedy min - mod')
# selected_elements_min_mod = []
# remaining_elements = set(distance_matrix.columns)
# for i in tqdm(range(len(distance_matrix.columns))):
#     next_monitor = find_min_modified_distance(distance_matrix, list(set(distance_matrix.columns)-set(remaining_elements)))
#     remaining_elements.remove(next_monitor)
#     selected_elements_min_mod.insert(0,next_monitor)

# proximity = {o_asn:MAX_LENGTH for o_asn in asn2asn.keys()}
# # proximity_min = [sum(proximity.values())]
# proximity_min_mod = []
# for i,e in tqdm(enumerate(selected_elements_min_mod)):
#     if i==0:
#         proximity = calculate_proximity(e, [], asn2asn, proximity)
#     else:
#         proximity = calculate_proximity(e, selected_elements_min_mod[0:i-1], asn2asn, proximity)
#     proximity_min_mod.append(sum(proximity.values()))
# proximity_min_mod = [(i-1)/INITIAL_PROXIMITY for i in proximity_min_mod]



print('Spectral Clustering')
df = dist_to_similarity_matrix(distance_matrix)
proximity_sp_clust = []
for k in NB_CLUSTERS:
    t = time.time()
    clustering = SpectralClustering(n_clusters=k, affinity='precomputed').fit(df.to_numpy())
    print(k, time.time()-t)
    current_proximity = []
    for i in range(10):
        set_monitors = samples_from_clusters(clustering.labels_, distance_matrix)
        proximity = {o_asn:MAX_LENGTH for o_asn in asn2asn.keys()}
        proximity = calculate_proximity_set(set_monitors, asn2asn, proximity)
        current_proximity.append(sum(proximity.values()))
    proximity_sp_clust.append(np.median(current_proximity))
    # proximity_sp_clust.append(sum(proximity.values()))
proximity_sp_clust = [(i-1)/INITIAL_PROXIMITY for i in proximity_sp_clust]



# print('Spectral Clustering 1')
# df = getAffinityMatrix(distance_matrix,k=20)
# proximity_sp_clust1 = []
# for k in NB_CLUSTERS:
#     t = time.time()
#     clustering = SpectralClustering(n_clusters=k, affinity='precomputed').fit(df.to_numpy())
#     print(k, time.time()-t)
#     current_proximity = []
#     for i in range(10):
#         set_monitors = samples_from_clusters(clustering.labels_, distance_matrix)
#         proximity = {o_asn:MAX_LENGTH for o_asn in asn2asn.keys()}
#         proximity = calculate_proximity_set(set_monitors, asn2asn, proximity)
#         current_proximity.append(sum(proximity.values()))
#     proximity_sp_clust1.append(np.median(current_proximity))
#     # proximity_sp_clust.append(sum(proximity.values()))
# proximity_sp_clust1 = [(i-1)/INITIAL_PROXIMITY for i in proximity_sp_clust1]

# print('tsne + Kmeans')
# df = dist_to_similarity_matrix(distance_matrix)
# df = df.to_numpy()
# df = TSNE(n_components=3).fit_transform(df)
# # df = PCA(n_components=0.9).fit_transform(df)

# proximity_kmeans = []
# for k in NB_CLUSTERS:
#     t = time.time()
#     clustering = KMeans(n_clusters=k).fit(df)
#     print(k, time.time()-t)
#     current_proximity = []
#     for i in range(10):
#         set_monitors = samples_from_clusters(clustering.labels_, distance_matrix)
#         proximity = {o_asn:MAX_LENGTH for o_asn in asn2asn.keys()}
#         proximity = calculate_proximity_set(set_monitors, asn2asn, proximity)
#         current_proximity.append(sum(proximity.values()))
#     proximity_kmeans.append(np.median(current_proximity))
# proximity_kmeans = [(i-1)/INITIAL_PROXIMITY for i in proximity_kmeans]





print('tsne + Kmeans - 10 clusters')
df = dist_to_similarity_matrix(distance_matrix)
df = df.to_numpy()
df = TSNE(n_components=3).fit_transform(df)
# df = PCA(n_components=0.9).fit_transform(df)


clustering = KMeans(n_clusters=10).fit(df)
proximity_kmeans = []
for i in range(3):
    set_monitors = samples_from_clusters_all_samples(clustering.labels_, distance_matrix)
    proximity = {o_asn:MAX_LENGTH for o_asn in asn2asn.keys()}
    current_proximity = []
    for i,e in tqdm(enumerate(set_monitors)):
        if i==0:
            proximity = calculate_proximity(e, [], asn2asn, proximity)
        else:
            proximity = calculate_proximity(e, set_monitors[0:i-1], asn2asn, proximity)
        current_proximity.append(sum(proximity.values()))
    proximity_kmeans.append(current_proximity)
proximity_kmeans = np.median(np.array(proximity_kmeans),axis=0)
proximity_kmeans = [(i-1)/INITIAL_PROXIMITY for i in proximity_kmeans]




with open('./data/proximities.json','r') as f:
    data = json.load(f)


proximity_opt = data['opt']
# proximity_sort = data['sort']
proximity_max = data['max']
proximity_min = data['min']
proximity_min_mod = data['min_mod']
# proximity_sp_clust = data['sp_clust']
# proximity_sp_clust1 = data['sp_clust1']
# proximity_kmeans = data['kmeans']

# with open('./data/proximities.json','w') as f:
#     json.dump({'opt':proximity_opt,'max':proximity_max,'min':proximity_min, 'min_mod':proximity_min_mod, 'sp_clust':proximity_sp_clust, 'kmeans':proximity_kmeans},f)




print('PLOTTING...')
#### PLOTS ####
fontsize = 15
fontsize_small = 7
fontsize_large = 15
linewidth = 2
markersize = 10

X = list(range(1,1+len(proximity_max)))
# plt.plot(X,proximity_max, X, proximity_min, X, proximity_opt, X, proximity_min_mod, NB_CLUSTERS, proximity_sp_clust, NB_CLUSTERS, proximity_kmeans)
plt.plot(X, proximity_opt, X,proximity_max, X, proximity_min, X, proximity_min_mod, NB_CLUSTERS, proximity_sp_clust, X, proximity_kmeans, linewidth=linewidth)
plt.legend(['Optimal','Greedy-max dist','Greedy-min dist', 'Greedy-min dist (mod)', 'Clustering (spectral)','Clustering (Kmeans)'], fontsize=fontsize)
plt.xlabel('#monitors', fontsize=fontsize)
plt.ylabel('Proximity (normalized)', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.tight_layout()
plt.grid(True)
plt.axis([1,1500,0.1,0.35])
plt.savefig('./figures/fig_proximity_vs_algorithm_similarity.png')
plt.xscale('log')
plt.savefig('./figures/fig_proximity_vs_algorithm_similarity_log.png')
plt.show()






