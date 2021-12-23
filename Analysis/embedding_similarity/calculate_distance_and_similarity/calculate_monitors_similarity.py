import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

DISTANCE_MATRIX_FNAME = 'RIPE_RIS_distance_embeddings_20211221.csv'


def scale_matrix(M, min_value=0.0, max_value=1.0):
    '''
    Scales the values of the given matrix to the interval [min_value, max_value], where the min_value/max_value correspond to
    the minimum/maximum value over all the elements in the input matrix
    :param  M:          (type: numpy array)
    :param  min_value:  (float) the minimum value of the scaled matrix; default 0.0
    :param  max_value:  (float) the maximum value of the scaled matrix; default 1.0
    :returns:           (type: numpy array) a scaled version of the input matrix M in the interval [min_value, max_value]
    '''

    min_max_scaler = MinMaxScaler(feature_range=(min_value, max_value))
    min_max_scaler.fit(M.flatten().reshape(-1, 1))

    return min_max_scaler.transform(M)


def dist_to_similarity_matrix(dist_matrix):
    '''
    Takes as input a distance matrix, whose element {i,j} denotes distance between the items in row {i} and column {j}
    (i.e., the larger the value the *less* similar are the items) and returns a similarity matrix, whose element {i,j}
    denotes similarity between the items {i} and {j} (i.e., the larger the value the *more* similar are the items).
    The following processing is applied:
        - Similarity is calculated as the inverse of the distance
        - The returned matrix is normalized to [0,1], where 1 corresponds to the highest similarity.
        - The nan values of the distance matrix are set to 0 (i.e., minimum similarity)
    :param dist_matrix: (type: pandas DataFrame) should be an NxN symmetric matrix (i.e., columns and index must be identical)
    :returns:           (type: pandas DataFrame) a similarity matrix
    '''

    # scale the distance matrix to [1,2] (to avoid divisions with zeros later)
    scaled_dist_matrix = scale_matrix(dist_matrix.to_numpy(copy=True), min_value=1, max_value=2)

    # set similarity to inverse of (scaled) distance, scale to [0,1],
    # and set NaN values to 0 (min similarity; i.e., equivalent to max_distance in the distance matrix)
    sim_matrix = 1 / scaled_dist_matrix
    sim_matrix = scale_matrix(sim_matrix)
    sim_matrix = np.nan_to_num(sim_matrix, nan=0)

    # transform the similarity matrix to pandas DataFrame with index/columns of the given matrix
    df = pd.DataFrame(sim_matrix)
    df.columns = dist_matrix.columns
    df.index = dist_matrix.index

    return df


def similarity_matrix_to_2D_vector(sim_matrix):
    '''
    Calculates from the input NxN matrix an Nx2 matrix where the each column corresponds to the {X,Y} coordinates
    of the tSNE transform. Elemements (i.e. rows) of the given matrix that are similar (i.e., have similar values
    in the same columns) are represented with coordinates with small distance.
    :param  sim_matrix: (type: pandas DataFrame) a similarity matrix NxN
    :returns:           (type: numpy array) a matrix Nx2
    '''
    return TSNE(n_components=2).fit_transform(sim_matrix.to_numpy())


print('Create 2D visualization dataset with rich data for RIPE RIS')
print('\t loading matrix ...')
with open(DISTANCE_MATRIX_FNAME, 'r') as f:
    distance_matrix = pd.read_csv(f, header=0, index_col=0)

similarity_matrix = dist_to_similarity_matrix(distance_matrix)
similarity_matrix.to_csv('RIPE_RIS_similarity_embeddings_20211221.csv', sep=',', index=True)

print('\t calculating tSNE vector ...')
df = similarity_matrix_to_2D_vector(similarity_matrix)


