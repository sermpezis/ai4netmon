from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import scipy as sc
import json

RIPE_RIS_PEERS = '../Datasets/RIPE_RIS_peers/improvements_RIPE_RIS_peers_leave_one_out.json'
PATH_AS_RELATIONSHIPS = '../Datasets/AS-relationships/20210701.as-rel2.txt'

DEEPWALK_EMBEDDINGS_128 = 'Embeddings/DeepWalk_128.csv'
DIFF2VEC_EMBEDDINGS_128 = 'Embeddings/Diff2Vec_128.csv'
NETMF_EMBEDDINGS_128 = 'Embeddings/NetMF_128.csv'
NODESKETCH_EMBEDDINGS_128 = 'Embeddings/NodeSketch_128.csv'
WALKLETS_EMBEDDINGS_256 = 'Embeddings/Walklets_256.csv'

NODE2VEC_EMBEDDINGS_64 = 'Embeddings/Node2Vec_embeddings.emb'
NODE2VEC_LOCAL_EMBEDDINGS_64 = 'Embeddings/Node2Vec_p2_64.csv'
NODE2VEC_GLOBAL_EMBEDDINGS_64 = 'Embeddings/Node2Vec_q2_64.csv'
DIFF2VEC_EMBEDDINGS_64 = 'Embeddings/Diff2Vec_64.csv'
NETMF_EMBEDDINGS_64 = 'Embeddings/NetMF_64.csv'
NODESKETCH_EMBEDDINGS_64 = 'Embeddings/NodeSketch_64.csv'
WALKLETS_EMBEDDINGS_128 = 'Embeddings/Walklets_128.csv'


def read_RIS_improvement_score():
    """
    :return: A dataframe containing 2 columns. The first one contains ASNs and the second one the improvement score
     that each ASN will bring to the Network
    """
    with open(RIPE_RIS_PEERS) as handle:
        dictdump = json.loads(handle.read())

    data = pd.DataFrame(dictdump.items(), columns=['ASN', 'Improvement_score'])

    data['ASN'] = data['ASN'].astype(np.int64)
    return data


def read_Node2Vec_embeddings_file():
    """
    :return: A dataframe containing the ASNs and the embeddings of each ASn created based on Node2Vec algorithm.
    """
    emb_df = pd.read_table(NODE2VEC_EMBEDDINGS_64, skiprows=1, header=None, sep=" ")
    # name the columns
    rng = range(0, 65)
    new_cols = ['dim_' + str(i) for i in rng]
    emb_df.columns = new_cols
    # rename first column
    emb_df.rename(columns={'dim_0': 'ASN'}, inplace=True)

    return emb_df


def read_karateClub_embeddings_file(emb, dimensions):
    """
    Karateclub library requires nodes to be named with consecutive Integer numbers. In the end gives as an output
    containing the embeddings in ascending order. So in this function we need to reassign each ASN to its own embedding.
    :param emb: A dataset containing pretrained embeddings
    :param dimensions: The dimensions of the given dataset
    :return: A dataframe containing pretrained embeddings
    """
    if dimensions == 64:
        if emb == 'Diff2Vec':
            df = pd.read_csv(DIFF2VEC_EMBEDDINGS_64, sep=',')
        elif emb == 'NetMF':
            df = pd.read_csv(NETMF_EMBEDDINGS_64, sep=',')
        elif emb == 'NodeSketch':
            df = pd.read_csv(NODESKETCH_EMBEDDINGS_64, sep=',')
        elif emb == 'Walklets':
            df = pd.read_csv(WALKLETS_EMBEDDINGS_128, sep=',')
        elif emb == 'Node2Vec_Local':
            df = pd.read_csv(NODE2VEC_LOCAL_EMBEDDINGS_64, sep=',')
        elif emb == 'Node2Vec_Global':
            df = pd.read_csv(NODE2VEC_GLOBAL_EMBEDDINGS_64, sep=',')
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
        elif emb == 'DeepWalk':
            df = pd.read_csv(DEEPWALK_EMBEDDINGS_128, sep=',')
        else:
            raise Exception('Not defined dataset')

    df['0'] = df['0'].astype(int)
    df.drop('0', axis=1, inplace=True)
    if emb == 'Walklets':
        dimensions = dimensions * 2
    else:
        dimensions = dimensions
    rng = range(1, dimensions)
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

    return final_df


def clear_dataset_from_outliers_z_score(data):
    """
    :param data: The merged dataframe created in merge_datasets function
    :return: A new dataframe where there are no outliers
    """
    # z-score may is not the right way to remove outliers for our dataset
    z_scores = sc.stats.zscore(data)
    abs_z_scores = np.abs(z_scores)
    filtered_entities = (abs_z_scores < 3).all(axis=1)
    new_df = data[filtered_entities]
    print(len(new_df))

    return new_df


def clear_dataset_from_outliers_inner_outer_fences(data):
    """
    :param data: The merged dataframe created in merge_datasets function
    :return: A dataframe that does not contain outliers
    """
    Q1 = data.Improvement_score.quantile(0.25)
    Q3 = data.Improvement_score.quantile(0.75)
    IQR = Q3 - Q1
    no_outliers = data.Improvement_score[
        (Q1 - 1.5 * IQR < data.Improvement_score) & (data.Improvement_score < Q3 + 1.5 * IQR)]
    outliers = data.Improvement_score[
        (Q1 - 1.5 * IQR >= data.Improvement_score) | (data.Improvement_score >= Q3 + 1.5 * IQR)]
    print(len(no_outliers))
    print(len(outliers))

    return no_outliers


def merge_datasets(improvement_df, embeddings_df):
    """
    :param improvement_df: Contains the improvement score that each ASN will bring to the network
    :param embeddings_df: Contains pretrained embeddings
    :return: A new merged dataset (containing improvement_score and the embedding of each ASN)
    """
    print(improvement_df['ASN'].isin(embeddings_df['ASN']).value_counts())
    mergedStuff = pd.merge(improvement_df, embeddings_df, on=['ASN'], how='left')
    mergedStuff.replace('', np.nan, inplace=True)
    mergedStuff = mergedStuff.dropna()
    outliers = 'Inner_Outer_Fences'
    if outliers == 'z_score':
        clear_df = clear_dataset_from_outliers_z_score(mergedStuff)
    elif outliers == 'Inner_Outer_Fences':
        clear_df = clear_dataset_from_outliers_inner_outer_fences(mergedStuff)

    return mergedStuff


def implement_pca(X):
    """
    :param X: The training set with the original number of features
    :return: The new training set with a smaller number of components that represent the 95% of variance
    """
    pca = PCA(n_components=0.95)
    X_new = pca.fit_transform(X)
    print("The number of components in order to keep variance in 95%: " + str(pca.n_components_))

    return X_new


def regression_stratify(y):
    """
    This function separates data into 2 bins in order to have always the same training and testing set.
    :param y: The label we want to predict
    :return:
    """
    min = np.amin(y)
    max = np.amax(y)

    bins = np.linspace(start=min, stop=max, num=2)
    y_binned = np.digitize(y, bins, right=True)
    return y_binned


def split_data_with_pca(X, y):
    """
    We need to implement first MinMaxScaler and after PCA
    :param data: The final dataframe
    :param X: The features that we give to our model in order to be trained
    :param y: The label we want to predict
    :return: Splits the data in x_train, x_test, y_train, y_test
    """
    scaler_choice = 'MinMaxScaler'
    if scaler_choice == 'MinMaxScaler':
        scaler_choice = MinMaxScaler(feature_range=(0, 1))
    elif scaler_choice == 'StandarScaler':
        scaler_choice = StandardScaler()
    X_scaled = scaler_choice.fit_transform(X)
    X_after_pca = implement_pca(X_scaled)
    y_binned = regression_stratify(y)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X_after_pca, y, test_size=0.1,
                                                                        stratify=y_binned, random_state=0)

    return x_train, x_test, y_train, y_test


def split_data(X, y):
    """
    :param X: The training set
    :param y: The label that we want to predict
    :return: Splits the data in x_train, x_test, y_train, y_test
    """
    scaler_choice = 'MinMaxScaler'
    if scaler_choice == 'MinMaxScaler':
        scaler_choice = MinMaxScaler(feature_range=(0, 1))
    elif scaler_choice == 'StandarScaler':
        scaler_choice = StandardScaler()
    X_scaled = scaler_choice.fit_transform(X)
    y_binned = regression_stratify(y)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X_scaled, y, test_size=0.1, stratify=y_binned,
                                                                        random_state=0)

    return x_train, x_test, y_train, y_test
