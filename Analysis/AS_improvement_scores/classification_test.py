import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import NearMiss
import matplotlib.pyplot as plt


def my_confusion_matrix(y_actual, y_predicted):

    """ This method finds the number of True Positives, False Positives,
    True Negatives and False Negative between the hidden movies
    and those predicted by the recommendation algorithm
    """
    cm = metrics.confusion_matrix(y_actual, y_predicted)
    return cm[0][0], cm[0][1], cm[1][1], cm[1][0]

def get_metrics(y_test, y_predicted):

    tp, fp, tn, fn = my_confusion_matrix(y_test, y_predicted)
    G_mean = np.sqrt((tp/(tp+fp)) * (tn/(tn+fp)))
    print('G-mean: %.4f' % G_mean)
    print('Balanced_Accuracy: %.4f' % metrics.balanced_accuracy_score(y_test, y_predicted))
    print('F1: %.4f' % metrics.f1_score(y_test, y_predicted, average="micro"))

def split_train_test(data, sampling=None):

    # Implement UnderSampling
    if sampling == 'undersample':
        dfs = []
        for i in range(0, 2):
            curr_df = data[data['top_k'] == i]
            dfs.append(resample(curr_df, replace=False, n_samples=1000, random_state=0))

        data = pd.concat(dfs)

    y = data['top_k']
    X = data.drop(['improvement_sc', 'iso', 'asn', 'source', 'longitude', 'latitude'], axis=1)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test

def Smote(X_train, y_train):

    smt = SMOTE(random_state=0)
    X_smote, y_smote = smt.fit_resample(X_train, y_train)
    return X_smote, y_smote

def Near_miss(X_train, y_train):

    undersample = NearMiss(version=1, n_neighbors=3)
    X_near, y_near = undersample.fit_resample(X_train, y_train)
    return X_near, y_near

def tomek_links(X_train, y_train):

    tl = TomekLinks()
    X_res, y_res = tl.fit_resample(X_train, y_train)
    return X_res, y_res

def Smotetomek(X_train, y_train):

    smt = SMOTETomek(random_state=0)
    X_smtm, y_smtm = smt.fit_resample(X_train, y_train)
    return X_smtm, y_smtm

def get_roc_auc(model, y_test, y_predicted):

    # Auc Curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predicted)
    auc = metrics.auc(fpr, tpr)
    roc_auc = metrics.roc_auc_score(y_test, y_predicted)
    # metrics.plot_roc_curve(model, x_test, y_test)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot(np.linspace(0, 1, 100),
             np.linspace(0, 1, 100),
             label='baseline',
             linestyle='--')
    plt.title('Receiver Operating Characteristic Curve', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.legend(fontsize=12)
    plt.show()


data = pd.read_csv('classification_task.csv', sep=",", dtype='unicode')
# Do not know how this column produced, so I drop it
data = data.drop('matched', axis=1)

# Need to convert column='improvement_sc' in order to apply nlargest function
data['improvement_sc'] = data['improvement_sc'].astype(str).astype(np.int64)
top_k = 1000
top_data = data.nlargest(top_k, 'improvement_sc')
print(top_data['improvement_sc'])

# print(top_data.improvement_sc.dtypes)
data['top_k'] = data['improvement_sc'].apply(lambda x: 1 if x in top_data['improvement_sc'].values else 0)
data.to_csv('papa.csv')
print(data.head())

data = data.fillna(0)

x_train, x_test, y_train, y_test = split_train_test(data)

method = 'Near_Miss'
if method == 'Smote':
    # Smote
    x_train, y_train = Smote(x_train, y_train)
elif method == 'Near_Miss':
    # Near_Miss
    x_train, y_train = Near_miss(x_train, y_train)
elif method == 'SmoteTomek':
    # Smote - Tomek
    x_train, y_train = Smotetomek(x_train, y_train)
else:
    x_train, x_test, y_train, y_test = split_train_test(data, 'undersample')

# # Decision Tree Classification
treeClassificationModel = DecisionTreeClassifier(random_state=0)
treeClassificationModel.fit(x_train, y_train)
y_predicted = treeClassificationModel.predict(x_test)
print("================= Decision Tree Regression =================")
get_metrics(y_test, y_predicted)
get_roc_auc(treeClassificationModel, y_test, y_predicted)

# Random Forest Classification
randomForestModel = RandomForestClassifier(random_state=0)
randomForestModel.fit(x_train, y_train)
y_predicted = randomForestModel.predict(x_test)
print("============ Random Forest Regression: =============")
get_metrics(y_test, y_predicted)
get_roc_auc(randomForestModel, y_test, y_predicted)

# XGBoost Classification
xgbClassification = XGBClassifier()
xgbClassification.fit(x_train, y_train)
y_predicted = xgbClassification.predict(x_test)
print("================ XGBoost Regression ================")
get_metrics(y_test, y_predicted)
get_roc_auc(xgbClassification, y_test, y_predicted)

# Multinomial Naive Bayes
clf = MultinomialNB()
clf.fit(x_train, y_train)
y_predicted = clf.predict(x_test)
print("============= Multinomial Naive Bayes ==============")
get_metrics(y_test, y_predicted)
get_roc_auc(clf, y_test, y_predicted)

# Logistic Regression
logreg = LogisticRegression(C=1e5, solver='lbfgs', max_iter=5000, multi_class='multinomial')
logreg.fit(x_train, y_train)
y_predicted = logreg.predict(x_test)
print("================ Logistic Regression: ================")
get_metrics(y_test, y_predicted)
get_roc_auc(logreg, y_test, y_predicted)