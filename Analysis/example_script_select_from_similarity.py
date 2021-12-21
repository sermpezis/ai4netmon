import pandas as pd
from ai4netmon.Analysis.similarity import similarity_utils as su
from ai4netmon.Analysis.similarity import select_from_similarity as sfs

## load a similarity matrix; e.g., use example 1 or 2 below
# example 1: toy example
items = ['a','b','c','d']
similarity_matrix = pd.DataFrame([[1.0,0.5,0.9,0.2],[0.5,1.0,0.3,0.5],[0.9,0.3,1.0,0.7],[0.2,0.5,0.7,1.0]], columns=items, index=items)
# example 2: load distance matrix and transform it to similarity matrix
DISTANCE_MATRIX_FNAME = '../data/similarity/ripe_ris_distance_pathlens_100k_20210701.csv'
distance_matrix = pd.read_csv(DISTANCE_MATRIX_FNAME, header=0, index_col=0)
similarity_matrix = su.dist_to_similarity_matrix(distance_matrix)

print('### Similarity Matrix ###')
print(similarity_matrix)
print()

# example 1: select based on method 'Greedy min'
selected_items1 = sfs.select_from_similarity_matrix(similarity_matrix, 'Greedy min')
print('Greedy min: first 4 selected items')
print(selected_items1[0:4])
print()

# example 2: select based on clustering method
kwargs = {'clustering_method':'Kmeans', 'nb_clusters':3}
selected_items2 = sfs.select_from_similarity_matrix(similarity_matrix, 'Clustering', **kwargs)
print('Clustering: first 4 selected items')
print(selected_items2[0:4])
print()