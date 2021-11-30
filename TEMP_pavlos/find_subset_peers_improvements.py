import json
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
from collections import defaultdict
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


SIMILARITY_MATRIX_FNAME = './data/similarity_matrix_distances_pathlens_100k.csv'
ASN2ASN_DIST_FNAME = './data/asn2asn__only_peers_pfx.json'
MAX_LENGTH = 20
NB_CLUSTERS = list(range(1,10))+[10, 20, 50, 100, 200]
INITIAL_PROXIMITY = 1443060#MAX_LENGTH*len(asn2asn.keys())

# define subset selection methods


def find_min_modified_distance(dist_matrix,removed_elements):
    remaining_elements = list(set(dist_matrix.columns) - set(removed_elements))
    df = dist_matrix.loc[remaining_elements,remaining_elements].copy()
    for p1 in df.columns:
        sorted_indexes = list(df.loc[p1,:].sort_values().index)
        df.loc[p1,sorted_indexes] = df.loc[p1,sorted_indexes] * list(range(1,1+len(df.columns))) 
    sum_distances = np.nansum(df, axis=0)
    if np.max(sum_distances) == 0: # all distances are nan or zero
        new_element = random.sample(remaining_elements, 1)[0]
    else:
        new_element = df.index[np.argmin(sum_distances)]
    return new_element


## Method: greedy largest distance
def find_max_distance(dist_matrix, selected_elements=[], start_with='max'):
    if (len(selected_elements) == 0) and (start_with == 'random'):
        new_element = random.sample(dist_matrix.columns, 1)
    else:
        not_selected_elements = list(set(dist_matrix.columns) - set(selected_elements))
        assert len(not_selected_elements) > 0, 'No more elements to be selected'        
        df = dist_matrix.loc[not_selected_elements,not_selected_elements]
        sum_distances = np.nansum(df, axis=0)
        if np.max(sum_distances) == 0: # all distances are nan or zero
            new_element = random.sample(not_selected_elements, 1)[0]
        else:
            new_element = df.index[np.argmax(sum_distances)]
    return new_element


## Method: greedy remove smallest distance
def find_min_distance(dist_matrix, removed_elements=[]):
    remaining_elements = list(set(dist_matrix.columns) - set(removed_elements))
    assert len(remaining_elements) > 0, 'No more elements to be selected'        
    df = dist_matrix.loc[remaining_elements,remaining_elements]
    sum_distances = np.nansum(df, axis=0)
    if np.max(sum_distances) == 0: # all distances are nan or zero
        new_element = random.sample(remaining_elements, 1)[0]
    else:
        new_element = df.index[np.argmin(sum_distances)]
    return new_element



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


def dist_to_similarity_matrix(dist_matrix):
    df = 1/dist_matrix
    df = df.replace(np.inf,np.nan)
    df = df.to_numpy()
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(df.flatten().reshape(-1, 1))
    df = min_max_scaler.transform(df)
    np.fill_diagonal(df, 1)
    np.nan_to_num(df,copy=False,nan=0)
    df = pd.DataFrame(df)
    df.columns = dist_matrix.columns
    df.index = dist_matrix.index
    return df



def getAffinityMatrix(dist_matrix, k = 7):
    df = dist_matrix.copy()
    df = df.replace(np.nan,np.inf)
    df = df.to_numpy()
    # for each row, sort the distances ascendingly and take the index of the 
    #k-th position (nearest neighbour)
    knn_distances = np.sort(df, axis=0)[k]
    knn_distances = knn_distances[np.newaxis].T
    
    # calculate sigma_i * sigma_j
    local_scale = knn_distances.dot(knn_distances.T)

    affinity_matrix = df * df
    affinity_matrix = -affinity_matrix / local_scale
    # divide square distance matrix by local scale
    affinity_matrix[np.where(np.isnan(affinity_matrix))] = 0.0
    # apply exponential
    affinity_matrix = np.exp(affinity_matrix)
    np.fill_diagonal(affinity_matrix, 0)

    df = pd.DataFrame(affinity_matrix)
    df.columns = dist_matrix.columns
    df.index = dist_matrix.index
    return df


def samples_from_clusters(cluster_labels, dist_matrix):
    cluster_members = defaultdict(list)
    for i, l in enumerate(cluster_labels):
        cluster_members[l].append(i)
    samples = []
    for l,m in cluster_members.items():
        ind = random.sample(m,1)[0]
        samples.append(dist_matrix.columns[ind])
    return samples

def samples_from_clusters_all_samples(cluster_labels, dist_matrix):
    cluster_members = defaultdict(list)
    for i, l in enumerate(cluster_labels):
        cluster_members[l].append(i)

    sorted_clusters = sorted(cluster_members, key=lambda k: len(cluster_members.get(k)), reverse=True)

    samples = []
    i = 0
    nb_clusters = len(sorted_clusters)
    while len(samples) < len(dist_matrix):
        ind = i%nb_clusters
        current_cluster = sorted_clusters[ind]
        len_current_cluster = len(cluster_members[current_cluster])
        if len_current_cluster > 0:
            s_ind = random.sample(range(len_current_cluster),1)[0]
            current_sample = cluster_members[current_cluster].pop(s_ind)
            samples.append(dist_matrix.columns[current_sample])
        else:
            if ind ==0:
                break
        i += 1
    return samples



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


# print('Sorted')
# proximity_peers = dict()
# for p in tqdm(distance_matrix.columns):
#     proximity_peers[p] = sum([dict_o_asn.get(p, MAX_LENGTH) for o_asn, dict_o_asn in asn2asn.items()])
# selected_elements_sort = sorted(proximity_peers, key=proximity_peers.get)

# proximity = {o_asn:MAX_LENGTH for o_asn in asn2asn.keys()}
# proximity_sort = []
# for i,e in tqdm(enumerate(selected_elements_sort)):
#     if i==0:
#         proximity = calculate_proximity(e, [], asn2asn, proximity)
#     else:
#         proximity = calculate_proximity(e, selected_elements_sort[0:i-1], asn2asn, proximity)
#     proximity_sort.append(sum(proximity.values()))


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






