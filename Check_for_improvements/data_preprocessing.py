from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import call_ML_models as cm
import pandas as pd
import numpy as np
import scipy as sc
import json
import smogn


FINAL_DATASET = '../Analysis/aggregate_data/asn_aggregate_data_20211201.csv'
RIPE_RIS_PEERS = '../Datasets/RIPE_RIS_peers/improvements_RIPE_RIS_peers_leave_one_out.json'
PATH_AS_RELATIONSHIPS = '../Datasets/AS-relationships/20210701.as-rel2.txt'
STUB_ASES = '../Analysis/remove_Stubs_from_AS_relationships/Stub_ASes.csv'

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
NODE2VEC_WL5_E3_LOCAL = 'Embeddings/Node2Vec_64_wl5_ws2_ep3_local.csv'
NODE2VEC_WL5_E3_GLOBAL = 'Embeddings/Node2Vec_64_wl5_ws2_ep3_global.csv'
NODE2VEC_64_WL5_E1_GLOBAL = 'Embeddings/Node2Vec_64_wl5_ws2_global.csv'
BGP2VEC_64 = 'Embeddings/Node2Vec_bgp2Vec.csv'
BGP2VEC_32 = 'Embeddings/BGP2VEC_32'


def call_smogn(final, flag_k_fold):
    smt = smogn.smoter(data=final, y='Improvement_score')
    y_smt = smt['Improvement_score']
    X_smt = smt.drop(['Improvement_score', 'ASN'], axis=1)
    x_train, x_test, y_train, y_test = split_data(X_smt, y_smt, flag_k_fold)
    x_train_pca, x_test_pca, y_train_pca, y_test_pca = split_data_with_pca(X_smt, np.log(y_smt), flag_k_fold)
    cm.call_methods_without_log(x_train, x_test, y_train, y_test)
    cm.call_methods_with_log(x_train_pca, x_test_pca, y_train_pca, y_test_pca)


def read_improvement_score():
    data = pd.read_csv("../Datasets/improvements20210601.txt", sep=" ")
    data.columns = ['location', 'IPV4-6', 'ASN', 'improvement_score']
    # keep only GLOBAL and IPV-4 examples
    new_data = data.loc[(data["location"] == "GLOBAL") & (data["IPV4-6"] == 4)]
    # GLOBAL 4 {21687,30104,46887} 1  --> 21687 1 || 30104 1 || 46887 1
    df_stack = pd.DataFrame(new_data.ASN.str.split(",").to_list(), index=new_data.improvement_score).stack()
    df_stack = df_stack.reset_index(["improvement_score"])
    df_stack.columns = ["Improvement_score", "ASN"]
    df_stack['ASN'] = df_stack['ASN'].str.strip('{}')
    df_stack = df_stack.reset_index(drop=True)
    df_stack['ASN'] = df_stack['ASN'].astype(str).astype(float)

    return df_stack


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


def read_final_dataset():
    return pd.read_csv(FINAL_DATASET)


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


def read_df_of_Stub_ASes():
    data = pd.read_csv(STUB_ASES)
    data.columns = ['ASN']
    data['ASN'] = data['ASN'].astype(float)

    return data


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
        elif emb == 'Node2Vec_wl5_global':
            df = pd.read_csv(NODE2VEC_64_WL5_E1_GLOBAL, sep=',')
        elif emb == 'Node2Vec_wl5_e3_global':
            df = pd.read_csv(NODE2VEC_WL5_E3_GLOBAL, sep=',')
        elif emb == 'Node2Vec_wl5_e3_local':
            df = pd.read_csv(NODE2VEC_WL5_E3_LOCAL, sep=',')
        elif emb == 'bgp2vec_64':
            df = pd.read_csv(BGP2VEC_64, sep=',')
        elif emb == 'bgp2vec_32':
            df = pd.read_csv(BGP2VEC_32, sep=',')
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
    if emb == 'Walklets':
        dimensions = dimensions * 2
    else:
        dimensions = 64
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

    return final_df


def convert_string_to_categorical(data):

    data.AS_rank_continent = pd.factorize(data.AS_rank_continent)[0]
    data.peeringDB_info_ratio = pd.factorize(data.peeringDB_info_ratio)[0]
    data.peeringDB_info_scope = pd.factorize(data.peeringDB_info_scope)[0]
    data.peeringDB_info_type = pd.factorize(data.peeringDB_info_type)[0]
    data.peeringDB_policy_general = pd.factorize(data.peeringDB_policy_general)[0]
    data.peeringDB_info_traffic.replace(np.nan, 0, inplace=True)
    data['peeringDB_info_traffic'] = data['peeringDB_info_traffic'].map({"0-20Mbps": 1, "20-100Mbps": 2, "100-1000Mbps": 3,
                                                                         "1-5Gbps": 4, "5-10Gbps": 5, "10-20Gbps": 6,
                                                                         "20-50Gbps": 7, "50-100Gbps": 8, "100-200Gbps": 9,
                                                                         "200-300Gbps": 10, "300-500Gbps": 11, "1-5Tbps": 12,
                                                                         "5-10Tbps": 13, "10-20Tbps": 14, "20-50Tbps": 15,
                                                                         "50-100Tbps": 16, "100+Tbps": 17, "500-1000Gbps": 18})
    data.replace(np.nan, 0, inplace=True)
    return data


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
    # print(len(new_df))

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
    no_outliers = no_outliers.to_frame()

    outliers = data.Improvement_score[
        (Q1 - 1.5 * IQR >= data.Improvement_score) | (data.Improvement_score >= Q3 + 1.5 * IQR)]
    # print(len(no_outliers))
    # print(len(outliers))

    new_df = pd.merge(no_outliers, data, left_index=True, right_index=True)
    new_df.drop(['Improvement_score_y', 'ASN'], axis=1, inplace=True)
    new_df.reset_index(inplace=True)
    new_df = new_df.rename(columns={'index': 'ASN', 'Improvement_score_x': 'Improvement_score'})

    return new_df


def merge_datasets(improvement_df, embeddings_df, final_df, z_score_flag, inner_outer_fences_flag):
    """
    :param inner_outer_fences_flag:  It shows us if we will use the tool (Inner_Outer_Fences)
    :param z_score_flag: It shows us if we will use the tool (Z-score)
    :param improvement_df: Contains the improvement score that each ASN will bring to the network
    :param embeddings_df: Contains pretrained embeddings
    :return: A new merged dataset (containing improvement_score and the embedding of each ASN)
    """
    print(improvement_df['ASN'].isin(embeddings_df['ASN']).value_counts())
    mergedStuff = pd.merge(improvement_df, embeddings_df, on=['ASN'], how='left')
    mergedStuff['ASN'] = mergedStuff['ASN'].astype(float)
    data = pd.merge(mergedStuff, final_df, on=['ASN'], how='left')
    data.replace('', np.nan, inplace=True)
    if z_score_flag:
        data_new = convert_string_to_categorical(data)
        df = data_new[data_new.T[data_new.dtypes != np.object].index]
        clear_df = clear_dataset_from_outliers_z_score(df)
    elif inner_outer_fences_flag:
        clear_df = clear_dataset_from_outliers_inner_outer_fences(data)
    else:
        clear_df = data

    return clear_df


def implement_pca(X):
    """
    :param X: The training set with the original number of features
    :return: The new training set with a smaller number of components that represent the 95% of variance
    """
    pca = PCA(n_components=0.95, svd_solver='full')
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


def split_data_with_pca(X, y, k_fold):
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
    if k_fold:
        new_X = pd.DataFrame(data=X_after_pca[1:, 1:], index=X_after_pca[1:, 0], columns=X_after_pca[0, 1:])
        k_fold_cross_validation(new_X, y, False)
    else:
        y_binned = regression_stratify(y)
        x_train, x_test, y_train, y_test = model_selection.train_test_split(X_after_pca, y, test_size=0.1,
                                                                            stratify=y_binned, random_state=0)
        return x_train, x_test, y_train, y_test


def split_data(X, y, k_fold):
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
    # X_after_pca = implement_pca(X_scaled)
    if k_fold:
        new_X = pd.DataFrame(data=X_scaled[1:, 1:], index=X_scaled[1:, 0], columns=X_scaled[0, 1:])
        k_fold_cross_validation(new_X, y, True)
    else:
        y_binned = regression_stratify(y)
        x_train, x_test, y_train, y_test = model_selection.train_test_split(X_scaled, y, test_size=0.1, random_state=0)
        return x_train, x_test, y_train, y_test


def k_fold_cross_validation(X, y, model_flag):
    """
    :param X: The training set
    :param y: The label that we want to predict
    :param model_flag: This flag helps us to give the correct input to Machine Learning models
    """
    kf = model_selection.KFold(n_splits=5, random_state=None)
    for train_index, test_index in kf.split(X):
        x_train, x_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        if model_flag:
            cm.call_methods_without_log(x_train, x_test, y_train, y_test)
        else:
            cm.call_methods_with_log(x_train, x_test, y_train, y_test)

    for i in cm.get_lists_containing_metrics():
        mse = np.mean([mse[0] for mse in i])
        mae = np.mean([mse[1] for mse in i])
        rmse = np.mean([mse[2] for mse in i])
        r2 = np.mean([mse[3] for mse in i])
        print(mse, mae, rmse, r2)


