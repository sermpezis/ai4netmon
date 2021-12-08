import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import similarity.similarity_utils as su


DISTANCE_MATRIX_FNAME = '../data/similarity/ripe_ris_distance_pathlens_100k_20210701.csv'
SIMILARITY_MATRIX_FNAME = '../data/similarity/ripe_atlas_probe_asn_similarity_jaccard_paths_v4_median75_asn_max_20211124.csv'


# case 1: distance matrix
print('Case 1: distance matrix - RIPE RIS')
print('\t loading matrix ...')
with open(DISTANCE_MATRIX_FNAME, 'r') as f:
  distance_matrix = pd.read_csv(f, header=0, index_col=0)
similarity_matrix = su.dist_to_similarity_matrix(distance_matrix)

print('\t calculating tSNE vector ...')
df = su.similarity_matrix_to_2D_vector(similarity_matrix)

print('\t plotting ...')
plt.scatter(df[:,0],df[:,1], s=3)
plt.savefig('fig_test_clustering_ripe_ris.png')
plt.close()





# case 2: similarity matrix
print('Case 2: similarity matrix - RIPE Atlas')
print('\t loading matrix ...')
with open(SIMILARITY_MATRIX_FNAME, 'r') as f:
  similarity_matrix = pd.read_csv(f, header=0, index_col=0)

# set empty values to minimum similarity (0) and self-similarity to max similarity (1)
similarity_matrix = similarity_matrix.replace(np.nan,0)
diagonal = tuple(np.arange(similarity_matrix.shape[0]))
similarity_matrix.values[diagonal,diagonal] = 1

print('\t calculating tSNE vector ...')
df = su.similarity_matrix_to_2D_vector(similarity_matrix)

print('\t plotting ...')
plt.scatter(df[:,0],df[:,1], s=3)
plt.savefig('fig_test_clustering_ripe_atlas.png')
plt.close()