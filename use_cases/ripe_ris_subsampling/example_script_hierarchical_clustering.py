import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from ai4netmon.Analysis.similarity import similarity_utils as su

ONLY_v4 = True
ONLY_v6 = False

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)



# load distance matrix and transform it to similarity matrix
DISTANCE_MATRIX_FNAME = '../../data/similarity/ripe_ris_distance_pathlens_100k_20210701.csv'
distance_matrix = pd.read_csv(DISTANCE_MATRIX_FNAME, header=0, index_col=0)
similarity_matrix = su.dist_to_similarity_matrix(distance_matrix)
distance_matrix = 1 - similarity_matrix

if ONLY_v4:
    peers_v4 = [m for m in distance_matrix.index if ':' not in m]
    distance_matrix = distance_matrix.loc[peers_v4, peers_v4]
elif ONLY_v6:
    peers_v6 = [m for m in distance_matrix.index if ':' in m]
    distance_matrix = distance_matrix.loc[peers_v6, peers_v6]


# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(affinity='precomputed', distance_threshold=0, n_clusters=None, linkage='average')
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model = model.fit(distance_matrix.to_numpy())

plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
# plot_dendrogram(model, truncate_mode="level", p=20, orientation='right', labels=distance_matrix.index)
# plot_dendrogram(model, truncate_mode="level", p=20, orientation='right', labels=distance_matrix.index, show_leaf_counts=False)
plot_dendrogram(model, orientation='right', labels=distance_matrix.index)
plt.savefig('fig_hierarchical_clustering_v4.png')
plt.show()
plt.close()