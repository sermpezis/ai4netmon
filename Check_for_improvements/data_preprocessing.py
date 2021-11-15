from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import json

RIPE_RIS_PEERS = '../Datasets/RIPE_RIS_peers/improvements_RIPE_RIS_peers_leave_one_out.json'
PATH_AS_RELATIONSHIPS = '../Datasets/AS-relationships/20210701.as-rel2.txt'

# NODE2VEC_EMBEDDINGS = 'Embeddings/Node2Vec_embeddings.emb'
DIFF2VEC_EMBEDDINGS_128 = 'Embeddings/Diff2Vec_128.csv'
NETMF_EMBEDDINGS_128 = 'Embeddings/NetMF_128.csv'
NODESKETCH_EMBEDDINGS_128 = 'Embeddings/NodeSketch_128.csv'
WALKLETS_EMBEDDINGS_256 = 'Embeddings/Walklets_256.csv'

NODE2VEC_EMBEDDINGS_64 = 'Embeddings/Node2Vec_embeddings.emb'
DIFF2VEC_EMBEDDINGS_64 = 'Embeddings/Diff2Vec_64.csv'
NETMF_EMBEDDINGS_64 = 'Embeddings/NetMF_64.csv'
NODESKETCH_EMBEDDINGS_64 = 'Embeddings/NodeSketch_64.csv'
WALKLETS_EMBEDDINGS_128 = 'Embeddings/Walklets_128.csv'

def read_RIS_improvement_score():
    with open(RIPE_RIS_PEERS) as handle:
        dictdump = json.loads(handle.read())

    data = pd.DataFrame(dictdump.items(), columns=['ASN', 'Improvement_score'])

    data['ASN'] = data['ASN'].astype(np.int64)
    return data


def read_Node2Vec_embeddings_file():
    emb_df = pd.read_table(NODE2VEC_EMBEDDINGS_64, skiprows=1, header=None, sep=" ")
    # name the columns
    rng = range(0, 65)
    new_cols = ['dim_' + str(i) for i in rng]
    emb_df.columns = new_cols
    # rename first column
    emb_df.rename(columns={'dim_0': 'ASN'}, inplace=True)

    return emb_df


def read_karateClub_embeddings_file(emb, dimensions):
    if dimensions == 64:
        if emb == 'Diff2Vec':
            df = pd.read_csv(DIFF2VEC_EMBEDDINGS_64, sep=',')
        elif emb == 'NetMF':
            df = pd.read_csv(NETMF_EMBEDDINGS_64, sep=',')
        elif emb == 'NodeSketch':
            df = pd.read_csv(NODESKETCH_EMBEDDINGS_64, sep=',')
        elif emb == 'Walklets':
            df = pd.read_csv(WALKLETS_EMBEDDINGS_128, sep=',')
        else:
            raise Exception('Not defined dataset')
    else:
        if emb == 'Diff2Vec':
            df = pd.read_csv(DIFF2VEC_EMBEDDINGS_128, sep=',')
        elif emb == 'NetMF':
            df = pd.read_csv(NETMF_EMBEDDINGS_128, sep=',')
        elif emb == 'NodeSketch':
            df = pd.read_csv(NODESKETCH_EMBEDDINGS_128, sep=',')
        elif emb == 'Walklets':
            df = pd.read_csv(WALKLETS_EMBEDDINGS_256, sep=',')
        else:
            raise Exception('Not defined dataset')

    df['0'] = df['0'].astype(int)
    if emb == 'Walklets':
        dimensions = dimensions*2
    else:
        dimensions = dimensions
    rng = range(1, dimensions + 1)
    other_cols = ['dim_' + str(i) for i in rng]
    first_col = ['ASN']
    new_cols = np.concatenate((first_col, other_cols), axis=0)
    df.columns = new_cols

    # Replace the consecutive ASNs given from KarateClub library with the initial ASNs
    data = pd.read_csv(PATH_AS_RELATIONSHIPS, sep="|", skiprows=180, header=None)
    data.columns = ['source', 'target', 'link', 'protocol']
    data.drop(['link', 'protocol'], axis=1, inplace=True)
    unique_nodes1 = set(data.source)
    unique_nodes2 = set(data.target)
    all_nodes = set(unique_nodes1.union(unique_nodes2))
    sort_nodes = sorted(all_nodes)
    previous_data = pd.DataFrame(sort_nodes)

    final_df = pd.concat([previous_data, df], axis=1)
    final_df.drop('ASN', axis=1, inplace=True)
    final_df.rename(columns={0: 'ASN'}, inplace=True)
    print(final_df)

    return final_df


def merge_datasets(improvement_df, embeddings_df):
    print(improvement_df['ASN'].isin(embeddings_df['ASN']).value_counts())

    mergedStuff = pd.merge(improvement_df, embeddings_df, on=['ASN'], how='left')
    mergedStuff.replace('', np.nan, inplace=True)
    mergedStuff = mergedStuff.dropna()

    return mergedStuff


def implement_pca(X):
    pca = PCA(n_components=10)
    X_new = pca.fit_transform(X)
    return X_new


def split_data_with_pca(X, y):
    # Implement PCA
    X = implement_pca(X)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test


def split_data(X, y):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test
