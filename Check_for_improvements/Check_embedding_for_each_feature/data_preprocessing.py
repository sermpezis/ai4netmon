from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import metrics


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
    """
    :param improvement_df: Contains the improvement score that each ASN will bring to the network
    :param embeddings_df: Contains pretrained embeddings
    :return: A new merged dataset (containing improvement_score and the embedding of each ASN)
    """
    print(improvement_df['ASN'].isin(embeddings_df['ASN']).value_counts())
    mergedStuff = pd.merge(embeddings_df, improvement_df, on=['ASN'], how='left')
    mergedStuff.replace('', 0, inplace=True)

    return mergedStuff


def split_data(X, y):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=0)

    return x_train, x_test, y_train, y_test


def get_regression_metrics(y_test, y_predicted):
    print("Mean Squared Error: %2f" % mean_squared_error(y_test, y_predicted))
    print("Mean Absolute Error: %2f" % mean_absolute_error(abs(y_test), abs(y_predicted)))
    print("RMSE: %2f" % np.sqrt(mean_squared_error(y_test, y_predicted)))
    print("R2 score: %2f" % r2_score(y_test, y_predicted))
    print("--------------------------")
    print()


def get_classification_metrics(y_test, y_predicted):
    tn, fp, tp, fn = my_confusion_matrix(y_test, y_predicted)
    print(tn, fp, tp, fn)
    G_mean = np.sqrt((tp / (tp + fp)) * (tn / (tn + fp)))
    print('G-mean: %.4f' % G_mean)
    print('Balanced_Accuracy: %.4f' % metrics.balanced_accuracy_score(y_test, y_predicted))
    print('F1: %.4f' % metrics.f1_score(y_test, y_predicted, average="micro"))


def call_regression_models(x_train, x_test, y_train, y_test):
    svRegressionModel = SVR(kernel="poly", max_iter=30000)
    svRegressionModel.fit(x_train, y_train)
    y_predicted = svRegressionModel.predict(x_test)
    print("Support Vector Regression: ")
    get_regression_metrics(y_test, y_predicted)

    dummy_reg = DummyRegressor(strategy="median")
    dummy_reg.fit(x_train, y_train)
    y_predicted = dummy_reg.predict(x_test)
    print("Dummy Regression: ")
    get_regression_metrics(y_test, y_predicted)


def my_confusion_matrix(y_actual, y_predicted):
    """ This method finds the number of True Negatives, False Positives,
    True Positives and False Negative between the hidden movies
    and those predicted by the recommendation algorithm
    """
    cm = metrics.confusion_matrix(y_actual, y_predicted)
    return cm[0][0], cm[0][1], cm[1][1], cm[1][0]


def change_string_class_to_categorical(data, feature):
    """
    This function convert strings to categorical values in order to proceed the classification procedure
    :param data: The given dataframe
    :param feature: The feature we want to convert to categorical
    :return: The feature with categorical values
    """
    le = preprocessing.LabelEncoder()
    le.fit(data[feature])
    data[feature] = le.transform(data[feature])

    return data


def call_classification_models(x_train, x_test, y_train, y_test):
    logreg = LogisticRegression(C=1e5, solver='saga', multi_class='multinomial', max_iter=1000)
    logreg.fit(x_train, y_train)
    y_predicted = logreg.predict(x_test)
    print("================ Logistic Regression: ================")
    get_classification_metrics(y_test, y_predicted)

    dummy_clf = DummyClassifier(strategy='most_frequent')
    dummy_clf.fit(x_train, y_train)
    y_predicted = dummy_clf.predict(x_test)
    print("================ Dummy Classifier: ================")
    get_classification_metrics(y_test, y_predicted)