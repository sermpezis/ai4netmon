from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

RIPE_RIS_PEERS = '../../Datasets/RIPE_RIS_peers/improvements_RIPE_RIS_peers_leave_one_out.json'
PATH_AS_RELATIONSHIPS = '../../Datasets/AS-relationships/20210701.as-rel2.txt'
STUB_ASES = '../../Analysis/remove_Stubs_from_AS_relationships/Stub_ASes.csv'

DEEPWALK_EMBEDDINGS_128 = '../Embeddings/DeepWalk_128.csv'
DIFF2VEC_EMBEDDINGS_128 = '../Embeddings/Diff2Vec_128.csv'
NETMF_EMBEDDINGS_128 = '../Embeddings/NetMF_128.csv'
NODESKETCH_EMBEDDINGS_128 = '../Embeddings/NodeSketch_128.csv'
WALKLETS_EMBEDDINGS_256 = '../Embeddings/Walklets_256.csv'

NODE2VEC_EMBEDDINGS_64 = '../Embeddings/Node2Vec_embeddings.emb'
NODE2VEC_LOCAL_EMBEDDINGS_64 = '../Embeddings/Node2Vec_p2_64.csv'
NODE2VEC_GLOBAL_EMBEDDINGS_64 = '../Embeddings/Node2Vec_q2_64.csv'
DIFF2VEC_EMBEDDINGS_64 = '../Embeddings/Diff2Vec_64.csv'
NETMF_EMBEDDINGS_64 = '../Embeddings/NetMF_64.csv'
NODESKETCH_EMBEDDINGS_64 = '../Embeddings/NodeSketch_64.csv'
WALKLETS_EMBEDDINGS_128 = '../Embeddings/Walklets_128.csv'
NODE2VEC_WL5_E3_LOCAL = '../Embeddings/Node2Vec_64_wl5_ws2_ep3_local.csv'
NODE2VEC_WL5_E3_GLOBAL = '../Embeddings/Node2Vec_64_wl5_ws2_ep3_global.csv'
NODE2VEC_64_WL5_E1_GLOBAL = '../Embeddings/Node2Vec_64_wl5_ws2_global.csv'
BGP2VEC_64 = '../Embeddings/Node2Vec_bgp2Vec.csv'
BGP2VEC_32 = '../Embeddings/BGP2VEC_32'

karate_club_emb_64 = ['Diff2Vec', 'NetMF', 'NodeSketch', 'Walklets', 'Node2Vec_Local', 'Node2Vec_Global',
                      'Node2Vec_wl5_global', 'Node2Vec_wl5_e3_global', 'Node2Vec_wl5_e3_local', 'bgp2vec_64',
                      'bgp2vec_32']
karate_club_emb_128 = ['Diff2Vec', 'NetMF', 'NodeSketch', 'Walklets', 'DeepWalk']
graph_emb_dimensions = 64


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
        dimensions = 32
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


def merge_datasets(improvement_df, embeddings_df):
    print(improvement_df['ASN'].isin(embeddings_df['ASN']).value_counts())
    mergedStuff = pd.merge(embeddings_df, improvement_df, on=['ASN'], how='left')
    mergedStuff.replace('', 0, inplace=True)
    # mergedStuff.replace(np.nan, 0.0, inplace=True)

    return mergedStuff


def split_data(X, y):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=0)

    return x_train, x_test, y_train, y_test


def get_metrics(y_test, y_predicted):
    print("Mean Squared Error: %2f" % mean_squared_error(y_test, y_predicted))
    print("Mean Absolute Error: %2f" % mean_absolute_error(abs(y_test), abs(y_predicted)))
    print("RMSE: %2f" % np.sqrt(mean_squared_error(y_test, y_predicted)))
    print("R2 score: %2f" % r2_score(y_test, y_predicted))
    print("--------------------------")
    print()


def call_regression_models(x_train, x_test, y_train, y_test):
    svRegressionModel = SVR(kernel="poly", max_iter=30000)
    svRegressionModel.fit(x_train, y_train)
    y_predicted = svRegressionModel.predict(x_test)
    print("Support Vector Regression: ")
    get_metrics(y_test, y_predicted)

    dummy_reg = DummyRegressor(strategy="median")
    dummy_reg.fit(x_train, y_train)
    y_predicted = dummy_reg.predict(x_test)
    print("Dummy Regression: ")
    get_metrics(y_test, y_predicted)


final_df = pd.read_csv('../../Analysis/aggregate_data/final_dataframe.csv')
embeddings_df = read_karateClub_embeddings_file(karate_club_emb_64[10], dimensions=graph_emb_dimensions)
embeddings_df['ASN'] = embeddings_df.ASN.astype(float)
mergedStuff = merge_datasets(final_df, embeddings_df)

# Fill the NaN values with the median value of the column
# mergedStuff['AS_hegemony'].fillna((mergedStuff['AS_hegemony'].mean()), inplace=True)
mergedStuff.dropna(subset=['AS_rank_rank'], inplace=True)
y = mergedStuff['AS_rank_rank']
X = mergedStuff.drop(
    ['ASN', 'AS_rank_rank', 'AS_rank_source', 'AS_rank_longitude', 'AS_rank_latitude', 'AS_rank_numberAsns',
     'AS_rank_numberPrefixes', 'AS_rank_numberAddresses', 'AS_rank_iso', 'AS_rank_total', 'AS_rank_customer',
     'AS_rank_peer', 'AS_rank_provider', 'is_personal_AS', 'peeringDB_info_ratio', 'peeringDB_info_traffic',
     'peeringDB_info_scope', 'peeringDB_info_type', 'peeringDB_info_prefixes4', 'peeringDB_info_prefixes6',
     'peeringDB_policy_general', 'peeringDB_ix_count', 'peeringDB_fac_count', 'peeringDB_created', 'AS_hegemony',
     'nb_atlas_probes_v4', 'nb_atlas_probes_v6'], axis=1)

x_train, x_test, y_train, y_test = split_data(X, y)
call_regression_models(x_train, x_test, y_train, y_test)
