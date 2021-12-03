from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection
import check_emb_with_regression as cewr
import numpy as np
import pandas as pd

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

karate_club_emb_64 = ['Diff2Vec', 'NetMF', 'NodeSketch', 'Walklets', 'Node2Vec_Local', 'Node2Vec_Global',
                      'Node2Vec_wl5_global', 'Node2Vec_wl5_e3_global', 'Node2Vec_wl5_e3_local', 'bgp2vec_64',
                      'bgp2vec_32']
karate_club_emb_128 = ['Diff2Vec', 'NetMF', 'NodeSketch', 'Walklets', 'DeepWalk']
graph_emb_dimensions = 64


def my_confusion_matrix(y_actual, y_predicted):
    """ This method finds the number of True Negatives, False Positives,
    True Positives and False Negative between the hidden movies
    and those predicted by the recommendation algorithm
    """
    cm = metrics.confusion_matrix(y_actual, y_predicted)
    return cm[0][0], cm[0][1], cm[1][1], cm[1][0]


def split_data(X, y):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=0)

    return x_train, x_test, y_train, y_test


def get_metrics(y_test, y_predicted):
    tn, fp, tp, fn = my_confusion_matrix(y_test, y_predicted)
    print(tn, fp, tp, fn)
    G_mean = np.sqrt((tp / (tp + fp)) * (tn / (tn + fp)))
    print('G-mean: %.4f' % G_mean)
    print('Balanced_Accuracy: %.4f' % metrics.balanced_accuracy_score(y_test, y_predicted))
    print('F1: %.4f' % metrics.f1_score(y_test, y_predicted, average="micro"))


def change_string_class_to_categorical(data, feature):
    le = preprocessing.LabelEncoder()
    le.fit(data.peeringDB_info_traffic)
    data[feature] = le.transform(data.peeringDB_info_traffic)

    return data


def call_classification_models(x_train, x_test, y_train, y_test):
    logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
    logreg.fit(x_train, y_train)
    y_predicted = logreg.predict(x_test)
    print("================ Logistic Regression: ================")
    get_metrics(y_test, y_predicted)

    dummy_clf = DummyClassifier(strategy='most_frequent')
    dummy_clf.fit(x_train, y_train)
    y_predicted = dummy_clf.predict(x_test)
    print("================ Dummy Classifier: ================")
    get_metrics(y_test, y_predicted)


final_df = pd.read_csv('../Analysis/aggregate_data/final_dataframe.csv')
embeddings_df = cewr.read_karateClub_embeddings_file(karate_club_emb_64[10], dimensions=graph_emb_dimensions)
embeddings_df['ASN'] = embeddings_df.ASN.astype(float)
mergedStuff = cewr.merge_datasets(final_df, embeddings_df)

mergedStuff.dropna(subset=['peeringDB_info_traffic'], inplace=True)
new_data = change_string_class_to_categorical(mergedStuff, 'peeringDB_info_traffic')
y = mergedStuff['peeringDB_info_traffic']
X = mergedStuff.drop(
    ['ASN', 'AS_rank_rank', 'AS_rank_source', 'AS_rank_longitude', 'AS_rank_latitude', 'AS_rank_numberAsns',
     'AS_rank_numberPrefixes', 'AS_rank_numberAddresses', 'AS_rank_iso', 'AS_rank_total', 'AS_rank_customer',
     'AS_rank_peer', 'AS_rank_provider', 'is_personal_AS', 'peeringDB_info_ratio', 'peeringDB_info_traffic',
     'peeringDB_info_scope', 'peeringDB_info_type', 'peeringDB_info_prefixes4', 'peeringDB_info_prefixes6',
     'peeringDB_policy_general', 'peeringDB_ix_count', 'peeringDB_fac_count', 'peeringDB_created', 'AS_hegemony',
     'nb_atlas_probes_v4', 'nb_atlas_probes_v6'], axis=1)

x_train, x_test, y_train, y_test = split_data(X, y)
call_classification_models(x_train, x_test, y_train, y_test)
