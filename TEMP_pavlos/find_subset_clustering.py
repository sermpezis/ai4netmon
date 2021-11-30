import json
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
from collections import defaultdict


from sklearn import preprocessing
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering

SIMILARITY_MATRIX_FNAME = './data/similarity_matrix_distances_pathlens_100k.csv'
ASN2ASN_DIST_FNAME = './data/asn2asn__only_peers_pfx.json'
MAX_LENGTH = 20



# load similarity matrix
print('Loading distance matrix')
with open(SIMILARITY_MATRIX_FNAME, 'r') as f:
  distance_matrix = pd.read_csv(f,header=0)
distance_matrix.index = distance_matrix.columns



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

def dist_to_dist_matrix(dist_matrix):
    df = dist_matrix.copy()
    m = df.max().max()
    df = df.replace(np.nan,m)
    df = df.to_numpy()
    np.fill_diagonal(df, 0)
    df = pd.DataFrame(df)
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








# df = dist_to_similarity_matrix(distance_matrix)
# # df = getAffinityMatrix(distance_matrix,k=20)



# NB_CLUSTERS = list(range(1,10))+[10, 20, 50, 100, 200]
# for k in NB_CLUSTERS:
#     t = time.time()
#     clustering = SpectralClustering(n_clusters=k, affinity='precomputed',n_components=20)
#     clustering.fit(df)
#     print(k, time.time()-t)
#     set_monitors = samples_from_clusters(clustering.labels_, distance_matrix)
#     print(list(pd.Series(clustering.labels_).value_counts()))





# df = dist_to_dist_matrix(distance_matrix)
# NB_CLUSTERS = list(range(1,10))+[10, 20, 50, 100, 200]
# for k in NB_CLUSTERS:
#     t = time.time()
#     clustering = DBSCAN(eps=1, min_samples=2, metric='precomputed')
#     clustering.fit(df.to_numpy())
#     print(k, time.time()-t)
#     print(pd.Series(clustering.labels_).value_counts())
#     set_monitors = samples_from_clusters(clustering.labels_, distance_matrix)
#     break




# from sklearn.decomposition import NMF
# df = dist_to_similarity_matrix(distance_matrix)
# NB_CLUSTERS = list(range(1,10))+[10, 20, 50, 100, 200]
# for k in NB_CLUSTERS:
#     t = time.time()
#     model = NMF(n_components=10)
#     W = model.fit_transform(df.to_numpy())
#     clustering = SpectralClustering(n_clusters=k)
#     clustering.fit(W)
#     print(k, time.time()-t)
#     set_monitors = samples_from_clusters(clustering.labels_, distance_matrix)
#     print(list(pd.Series(clustering.labels_).value_counts()))






from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
df = dist_to_similarity_matrix(distance_matrix)
df = df.to_numpy()
df = TSNE(n_components=2).fit_transform(df)
# df = PCA(n_components=0.8).fit_transform(df)
plt.scatter(df[:,0],df[:,1], s=3)
plt.savefig('./figures/fig_tsne_clustering.png')
plt.show()


NB_CLUSTERS = list(range(1,10))+[10, 20, 50, 100, 200]
for k in NB_CLUSTERS:
    t = time.time()
    clustering = KMeans(n_clusters=k)
    clustering.fit(df)
    print(k, time.time()-t)
    set_monitors = samples_from_clusters(clustering.labels_, distance_matrix)
    print(list(pd.Series(clustering.labels_).value_counts()))


