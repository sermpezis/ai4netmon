import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import nan_euclidean_distances
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib import pyplot as plt
import json


SAMPLES_FNAME = './data/samples_pathlens_100000.csv'
SIMILARITY_MATRIX_FNAME = './data/similarity_matrix_distances_pathlens_100k.csv'
RIPE_RIS_PEERS_FNAME = 'list_of_RIPE_RIS_peers.json'
PATHLEN_DISTANCES_SUM_FNAME = './data/pathlen_distances_sum.json'
PATHLEN_DISTANCES_AVG_FNAME = './data/pathlen_distances_avg.json'
PATHLEN_DISTANCES_MIN_FNAME = './data/pathlen_distances_min.json'
PATHLEN_DISTANCES_MAX_FNAME = './data/pathlen_distances_max.json'
PATHLEN_DISTANCES_SUM_5MAX_FNAME = './data/pathlen_distances_sum_5max.json'
PATHLEN_DISTANCES_SUM_10MAX_FNAME = './data/pathlen_distances_sum_10max.json'
PATHLEN_DISTANCES_SUM_5MIN_FNAME = './data/pathlen_distances_sum_5min.json'
PATHLEN_DISTANCES_SUM_10MIN_FNAME = './data/pathlen_distances_sum_10min.json'
PATHLEN_DISTANCES_SUM_5MAX_FNAME = './data/pathlen_distances_sum_5max.json'
PATHLEN_DISTANCES_SUM_10MAX_FNAME = './data/pathlen_distances_sum_10max.json'
PATHLEN_DISTANCES_SUM_5MIN_FNAME = './data/pathlen_distances_sum_5min.json'
PATHLEN_DISTANCES_SUM_10MIN_FNAME = './data/pathlen_distances_sum_10min.json'
THRESHOLD_MIN_SAMPLES = 200

normalized_nan_euclidean_distances = lambda x : nan_euclidean_distances(x)/x.shape[1]
adapted_normalized_nan_euclidean_distances = lambda x,max_size : nan_euclidean_distances(x)/ max_size *np.sqrt(max_size/x.shape[1])



print('Reading csv file with pathlens...')
df = pd.read_csv(SAMPLES_FNAME, delimiter=',')
print('Basic data handling ...')
df = df.replace(-1,np.nan)
df = df.dropna(axis=0,how='all')
df = df.dropna(axis=1,how='all')

# df1 = df.iloc[0:3,0:3]
# # df1 = df1.to_numpy().transpose()
# df2 = df.iloc[0:2,0:3]
# # df2 = df2.to_numpy().transpose()
# print(df1)
# print(df1.shape)

print('Calculating transpose ...')
a = df.to_numpy().transpose()


# avg_distance = [1]
# max_distance = [1]
# previous_distance_matrix = np.ones((a.shape[0],a.shape[0]))
# for i in range(1,a.shape[1]):
#     distance_matrix = normalized_nan_euclidean_distances(a[:,0:i])
#     diff = np.abs(distance_matrix-previous_distance_matrix)
#     avg_distance.append(np.nanmean(diff))
#     max_distance.append(np.nanmax(diff))
#     previous_distance_matrix = distance_matrix.copy()
#     np.fill_diagonal(distance_matrix,np.nan)
#     # print(distance_matrix)
#     perfect_sim = np.where(distance_matrix==0)
#     print(len(perfect_sim[0]))


# plt.plot(range(a.shape[1]),max_distance,range(a.shape[1]),avg_distance)
# plt.yscale('log')
# plt.show()


print('Calculating Euclidean distances...')
distance_matrix = normalized_nan_euclidean_distances(a)
np.fill_diagonal(distance_matrix,np.nan)

print('Creating the distance matrix...')
df[~df.isna()] = 1
df[df.isna()] = 0
u = df.to_numpy()
common_samples = np.matmul(u.transpose(),u)
distance_matrix[common_samples<THRESHOLD_MIN_SAMPLES] = np.nan


print('Saving the distance matrix...')
pd.DataFrame(distance_matrix, columns=df.columns).to_csv(SIMILARITY_MATRIX_FNAME, index=False)



print('Calculating distance metrics...')
with open(RIPE_RIS_PEERS_FNAME, 'r') as f:
    ripe_ris_peers = json.load(f)

sum_distances = np.nansum(distance_matrix, axis=0)
sum_distances[np.isnan(distance_matrix).all(axis=0)] = np.nan
avg_distances = np.nanmean(distance_matrix, axis=0)
min_distances = np.nanmin(distance_matrix, axis=0)
max_distances = np.nanmax(distance_matrix, axis=0)
total_distance = {df.columns[i]: sum_distances[i] for i in range(sum_distances.shape[0])}
avg_distance = {df.columns[i]: avg_distances[i] for i in range(avg_distances.shape[0])}
min_distance = {df.columns[i]: min_distances[i] for i in range(min_distances.shape[0])}
max_distance = {df.columns[i]: max_distances[i] for i in range(max_distances.shape[0])}


nan_nansum = lambda x: np.nan if np.isnan(x).all() else np.nansum(x)

sum_5max_distances = {df.columns[i]:nan_nansum(np.sort(distance_matrix[i,:])[0:5]) for i in range(sum_distances.shape[0])}
sum_10max_distances = {df.columns[i]:nan_nansum(np.sort(distance_matrix[i,:])[0:10]) for i in range(sum_distances.shape[0])}
sum_5min_distances = {df.columns[i]:nan_nansum(np.sort(distance_matrix[i,~np.isnan(distance_matrix[i,:])])[::-1][0:5]) for i in range(sum_distances.shape[0])}
sum_10min_distances = {df.columns[i]:nan_nansum(np.sort(distance_matrix[i,~np.isnan(distance_matrix[i,:])])[::-1][0:10]) for i in range(sum_distances.shape[0])}


with open(PATHLEN_DISTANCES_SUM_FNAME,'w') as f:
    json.dump(total_distance, f)
with open(PATHLEN_DISTANCES_AVG_FNAME,'w') as f:
    json.dump(avg_distance, f)
with open(PATHLEN_DISTANCES_MIN_FNAME,'w') as f:
    json.dump(min_distance, f)
with open(PATHLEN_DISTANCES_MAX_FNAME,'w') as f:
    json.dump(max_distance, f)
with open(PATHLEN_DISTANCES_SUM_5MAX_FNAME,'w') as f:
    json.dump(sum_5max_distances, f)
with open(PATHLEN_DISTANCES_SUM_10MAX_FNAME,'w') as f:
    json.dump(sum_10max_distances, f)
with open(PATHLEN_DISTANCES_SUM_5MIN_FNAME,'w') as f:
    json.dump(sum_5min_distances, f)
with open(PATHLEN_DISTANCES_SUM_10MIN_FNAME,'w') as f:
    json.dump(sum_10min_distances, f)


# list_of_perfect_sims = []
# for i in range(len(perfect_sim[0])):
#     ind1 = perfect_sim[0][i]
#     ind2 = perfect_sim[1][i]
#     if ind1 < ind2:
#         peer_ip1 = df.columns[ind1]
#         peer_ip2 = df.columns[ind2]
#         peer_asn1 = ripe_ris_peers[peer_ip1]
#         peer_asn2 = ripe_ris_peers[peer_ip2]
#         if (peer_asn1 != peer_asn2) and (sum(df.iloc[:,ind1].notna() & df.iloc[:,ind2].notna())>10):
#             print([ind1, ind2, peer_ip1, peer_ip2, peer_asn1, peer_asn2, sum(df.iloc[:,ind1].notna() & df.iloc[:,ind2].notna())])


