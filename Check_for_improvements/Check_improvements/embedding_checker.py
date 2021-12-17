from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

PATH_AS_RELATIONSHIPS = '../../Datasets/AS-relationships/20210701.as-rel2.txt'

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

NODE2VEC_32_WL6_WN40_EP3 = '../Embeddings/Node2Vec_32_wl6_ws5_ep3_wn40_p2_q05.csv'
BGP2VEC_32 = '../Embeddings/BGP2VEC_32'
BGP2VEC_32_WS5 = '../Embeddings/BGP2Vec_32_wl6_ws5_ep3_wn40_p4_q05.csv'

karate_club_emb_64 = ['Diff2Vec', 'NetMF', 'NodeSketch', 'Walklets', 'Node2Vec_Local', 'Node2Vec_Global', 'Node2Vec_wl5_global', 'Node2Vec_wl5_e3_global', 'Node2Vec_wl5_e3_local', 'bgp2vec_64', 'bgp2vec_32', 'bgp2vec_32_ws5', 'node2vec_32_wl6_wn40_e3']
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
        elif emb == 'bgp2vec_32_ws5':
            df = pd.read_csv(BGP2VEC_32_WS5, sep=',')
        elif emb == 'node2vec_32_wl6_wn40_e3':
            df = pd.read_csv(BGP2VEC_32_WS5, sep=',')
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
    """
    :param improvement_df: Contains the improvement score that each ASN will bring to the network
    :param embeddings_df: Contains pretrained embeddings
    :return: A new merged dataset (containing improvement_score and the embedding of each ASN)
    """
    print(improvement_df['ASN'].isin(embeddings_df['ASN']).value_counts())
    mergedStuff = pd.merge(improvement_df, embeddings_df, on=['ASN'], how='left')
    mergedStuff.replace('', np.nan, inplace=True)
    mergedStuff = mergedStuff.dropna()

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


def call_methods(x_train, x_test, y_train, y_test):

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


data = pd.read_csv("../../Datasets/improvements20210601.txt", sep=" ")
data.columns = ['location', 'IPV4-6', 'ASN', 'improvement_score']
# keep only GLOBAL and IPV-4 examples
new_data = data.loc[(data["location"] == "GLOBAL") & (data["IPV4-6"] == 4)]
# GLOBAL 4 {21687,30104,46887} 1  --> 21687 1 || 30104 1 || 46887 1
df_stack = pd.DataFrame(new_data.ASN.str.split(",").to_list(), index=new_data.improvement_score).stack()
df_stack = df_stack.reset_index(["improvement_score"])
df_stack.columns = ["improvement_score", "ASN"]
df_stack['ASN'] = df_stack['ASN'].str.strip('{}')
df_stack = df_stack.reset_index(drop=True)
df_stack['ASN'] = df_stack['ASN'].astype(str).astype(float)


embeddings_df = read_karateClub_embeddings_file(karate_club_emb_64[12], dimensions=graph_emb_dimensions)
embeddings_df['ASN'] = embeddings_df.ASN.astype(float)
mergedStuff = merge_datasets(df_stack, embeddings_df)
print(mergedStuff)
print(mergedStuff.isnull().values.any())

y = mergedStuff['improvement_score']
print(y)
X = mergedStuff.drop(['ASN', 'improvement_score'], axis=1)

x_train, x_test, y_train, y_test = split_data(X, y)
call_methods(x_train, x_test, y_train, y_test)
