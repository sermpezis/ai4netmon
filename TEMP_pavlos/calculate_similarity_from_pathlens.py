import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import nan_euclidean_distances
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib import pyplot as plt
import json


SAMPLES_FNAME = 'samples_pathlens.csv'
RIPE_RIS_PEERS_FNAME = 'list_of_RIPE_RIS_peers.json'
PATHLEN_SIMILARITIES_SUM_FNAME = 'pathlen_similarities_sum.json'
PATHLEN_SIMILARITIES_MIN_FNAME = 'pathlen_similarities_min.json'
THRESHOLD_MIN_SAMPLES = 200

normalized_nan_euclidean_distances = lambda x : nan_euclidean_distances(x)/x.shape[1]
adapted_normalized_nan_euclidean_distances = lambda x,max_size : nan_euclidean_distances(x)/ max_size *np.sqrt(max_size/x.shape[1])




df = pd.read_csv(SAMPLES_FNAME, delimiter=',')
df = df.replace(-1,np.nan)
df = df.dropna(axis=0,how='all')
df = df.dropna(axis=1,how='all')

df1 = df.iloc[0:3,0:3]
# df1 = df1.to_numpy().transpose()
df2 = df.iloc[0:2,0:3]
# df2 = df2.to_numpy().transpose()
print(df1)
print(df1.shape)

# df = df1
a = df.to_numpy().transpose()


avg_distance = [1]
max_distance = [1]
previous_similarity_matrix = np.ones((a.shape[0],a.shape[0]))
for i in range(1,a.shape[1]):
    similarity_matrix = normalized_nan_euclidean_distances(a[:,0:i])
    diff = np.abs(similarity_matrix-previous_similarity_matrix)
    avg_distance.append(np.nanmean(diff))
    max_distance.append(np.nanmax(diff))
    previous_similarity_matrix = similarity_matrix.copy()
    np.fill_diagonal(similarity_matrix,np.nan)
    # print(similarity_matrix)
    perfect_sim = np.where(similarity_matrix==0)
    print(len(perfect_sim[0]))


plt.plot(range(a.shape[1]),max_distance,range(a.shape[1]),avg_distance)
plt.yscale('log')
plt.show()



for i in range(similarity_matrix.shape[1]):
    print(i)
    for j in range(similarity_matrix.shape[1]):
        if (sum(df.iloc[:,i].notna() & df.iloc[:,j].notna())<THRESHOLD_MIN_SAMPLES):
            similarity_matrix[i,j] = np.nan



with open(RIPE_RIS_PEERS_FNAME, 'r') as f:
    ripe_ris_peers = json.load(f)

sum_similarities = np.nansum(similarity_matrix, axis=0)
min_similarities = np.nanmin(similarity_matrix, axis=0)
total_distance = {df.columns[i]: sum_similarities[i] for i in range(sum_similarities.shape[0])}
min_distance = {df.columns[i]: min_similarities[i] for i in range(min_similarities.shape[0])}

with open(PATHLEN_SIMILARITIES_SUM_FNAME,'w') as f:
    json.dump(total_distance, f)
with open(PATHLEN_SIMILARITIES_MIN_FNAME,'w') as f:
    json.dump(min_distance, f)


list_of_perfect_sims = []
for i in range(len(perfect_sim[0])):
    ind1 = perfect_sim[0][i]
    ind2 = perfect_sim[1][i]
    if ind1 < ind2:
        peer_ip1 = df.columns[ind1]
        peer_ip2 = df.columns[ind2]
        peer_asn1 = ripe_ris_peers[peer_ip1]
        peer_asn2 = ripe_ris_peers[peer_ip2]
        if (peer_asn1 != peer_asn2) and (sum(df.iloc[:,ind1].notna() & df.iloc[:,ind2].notna())>10):
            print([ind1, ind2, peer_ip1, peer_ip2, peer_asn1, peer_asn2, sum(df.iloc[:,ind1].notna() & df.iloc[:,ind2].notna())])


