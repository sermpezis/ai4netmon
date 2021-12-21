import pandas as pd
import json
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from ai4netmon.Analysis.similarity import similarity_utils as su
from ai4netmon.Analysis.similarity import select_from_similarity as sfs
from collections import defaultdict


MAX_LENGTH = 20
INITIAL_PROXIMITY = 1443060#MAX_LENGTH*len(asn2asn.keys())
ASN2ASN_DIST_FNAME = '../../data/misc/asn2asn__only_peers_pfx.json'
PROXIMITY_FNAME = 'proximity_selected_monitors_ripe_ris_pathlens_100k.json'

'''
Auxiliary methods used for calculating the target objective "proximity". The proximity quantifies 
how close to origin ASes a selected set of RIPE RIS peers are. A peer X seeing an AS path of length 5 
to the origin AS Y, has proximity(X,Y)=5. If another peer Z has proximity(Z,Y)=3, then proximity({X,Z},Y)=3. 
The default value of max proximity (i.e., no visibility of the origin AS) is set to MAX_LENGTH. The lower the
value of the proximity to all origin ASes, the better for the monitoring system.
'''
def calculate_proximity(next_item, asn2asn, proximity):
    for o_asn, dict_o_asn in asn2asn.items():
        if (next_item in dict_o_asn.keys()) and (dict_o_asn[next_item] < proximity[o_asn]):
            proximity[o_asn] = dict_o_asn[next_item]
    return proximity

def get_proximity_vector(selected_items, asn2asn):
    proximity = {o_asn:MAX_LENGTH for o_asn in asn2asn.keys()}
    proximity_vector = []
    for i, item in tqdm(enumerate(selected_items)):
        proximity = calculate_proximity(item, asn2asn, proximity)
        proximity_vector.append(sum(proximity.values()))
    proximity_vector = [(i-1)/INITIAL_PROXIMITY for i in proximity_vector]  # normalize proximity to [0,1]
    return proximity_vector



# load proximity victionary {originAS1: {peer1:proximity11, peer2: proximity12, ...}, originAS2: ...}
print('Loading proximity dict...')
with open(ASN2ASN_DIST_FNAME, 'r') as f:
    asn2asn = json.load(f)

# find full feeding peers
feed = defaultdict(lambda : 0)
for o_asn, dict_o_asn in asn2asn.items():
    for m_asn, dist in dict_o_asn.items():
        feed[m_asn] +=1
full_feeders = [m_asn for m_asn, nb_feeds in feed.items() if nb_feeds > 65000]



# items = ['a','b','c','d']
# similarity_matrix = pd.DataFrame([[1.0,0.5,0.9,0.2],[0.5,1.0,0.3,0.5],[0.9,0.3,1.0,0.7],[0.2,0.5,0.7,1.0]], columns=items, index=items)

# load distance matrix and transform it to similarity matrix
DISTANCE_MATRIX_FNAME = '../data/similarity/ripe_ris_distance_pathlens_100k_20210701.csv'
distance_matrix = pd.read_csv(DISTANCE_MATRIX_FNAME, header=0, index_col=0)
similarity_matrix = su.dist_to_similarity_matrix(distance_matrix)


print('### Similarity matrix ###')
print(similarity_matrix)
print()

# define what methods will be used
## the "Greedy min-mod" method takes a lot of time (e.g., ~20min for ~1300x1300 similarity matrix; the other methods take approx 5sec or less)
method_param_dict = {
    'Greedy min':    {'method':'Greedy min', 'sim_matrix': similarity_matrix, 'args':{}},
    'Greedy min full':    {'method':'Greedy min', 'sim_matrix': similarity_matrix.loc[full_feeders,full_feeders], 'args':{}},
    # 'Greedy min mod':{'method':'Greedy min', 'sim_matrix': similarity_matrix, 'args':{'rank_normalization':True} },
    'Greedy max':    {'method':'Greedy max', 'sim_matrix': similarity_matrix, 'args':{}},
    'Greedy max full':    {'method':'Greedy max', 'sim_matrix': similarity_matrix.loc[full_feeders,full_feeders], 'args':{}},
    'Clustering spectral k10':    {'method':'Clustering', 'sim_matrix': similarity_matrix, 'args':{'clustering_method':'SpectralClustering', 'nb_clusters':10}},
    'Clustering spectral k10 full':    {'method':'Clustering', 'sim_matrix': similarity_matrix.loc[full_feeders,full_feeders], 'args':{'clustering_method':'SpectralClustering', 'nb_clusters':10}},
    'Clustering kmeans k10':      {'method':'Clustering', 'sim_matrix': similarity_matrix, 'args':{'clustering_method':'Kmeans', 'nb_clusters':10}},
    'Clustering kmeans k10 full':      {'method':'Clustering', 'sim_matrix': similarity_matrix.loc[full_feeders,full_feeders], 'args':{'clustering_method':'Kmeans', 'nb_clusters':10}}
}


print('### Selected monitors by method ###')
for m, params in method_param_dict.items():
    selected_items = sfs.select_from_similarity_matrix(params['sim_matrix'], params['method'], **params['args'])
    print('\t{} [DONE]'.format(m))
    with open('dataset_selected_monitors_ripe_ris_pathlens_100k_{}.json'.format('_'.join(m.lower().translate('()').split(' '))), 'w') as f:
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
        with open('dataset_selected_monitors_ripe_ris_pathlens_100k_{}.json'.format('_'.join(m.lower().split(' '))), 'r') as f:
            selected_items = json.load(f)
        proximity_vector[m] = get_proximity_vector(selected_items, asn2asn)
        print('\t{} [DONE]'.format(m))
    with open(PROXIMITY_FNAME, 'w') as f:
        json.dump(proximity_vector, f)



print('PLOTTING...')
fontsize = 15
fontsize_small = 7
fontsize_large = 15
linewidth = 2
markersize = 10


colors = ['g','--g','r','--r','b','--b','k','--k']
leg_str = []
for i, k in enumerate(proximity_vector.keys()):
    print(k, proximity_vector[k][100])
    X = list(range(1,1+len(proximity_vector[k])))
    plt.plot(X, proximity_vector[k], colors[i], linewidth=linewidth)
    leg_str.append(k)
plt.xscale('log')
plt.xlabel('#monitors', fontsize=fontsize)
plt.ylabel('Proximity (normalized)', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.tight_layout()
plt.axis([1,1500,0.1,0.35])
plt.legend(leg_str)
plt.grid(True)
plt.savefig('fig_ripe_ris_subset_selection_vs_proximity.png')
# plt.show()





