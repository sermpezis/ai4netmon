import numpy as np
import pandas as pd
import seaborn as sns
import os.path
import re
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF
import json
import csv
import models_metrics as models
from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def keep_number(text):

    num = re.findall(r'[0-9]+', text)
    return " ".join(num)

def exists_in_perso(perso, asn):

    return asn in perso.asn.values

def implement_pca(X):

    pca = PCA()
    X_new = pca.fit_transform(X)
    return X_new

def preprocess():

    if not os.path.isfile('../../Datasets/As-rank/asns.csv'):
        # Opening JSON file and loading the data
        # into the variable data
        with open('../../Datasets/As-rank/asns.json') as json_file:
            jsondata = json.load(json_file)

        # employee_data = data['asn', 'rank', 'source', 'longitude', 'latitude', 'numberAsns', 'numberPrefixes', 'numberAddresses', 'iso', 'total', 'customer', 'peer', 'provider']
        data_file = open('../../Datasets/As-rank/asns.csv', 'w', newline='')

        csv_writer = csv.writer(data_file)

        count = 0
        for data in jsondata:
            if count == 0:
                header = data.keys()
                csv_writer.writerow(header)
                count += 1
            csv_writer.writerow(data.values())

        data_file.close()
    else:
        # checking data before we start preprocessing
        data = pd.read_csv('../../Datasets/As-rank/asns.csv', sep=",")
        print(data.head())
        print(data.shape)
        print(data.dtypes)
        print("------------------------------")
    return data

def split_data(X, y):

    # Implement PCA
    # X = implement_pca(X)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=0)
    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test


data = preprocess()
data2 = pd.read_csv("../../Datasets/improvements20210601.txt", sep=" ", header=None)
# Give name to columns
data2.columns = ['location', 'IPV4-6', 'asn', 'improvement_sc']

# keep only GLOBAL and IPV-4 examples
dt = data2.loc[(data2["location"] == "GLOBAL") & (data2["IPV4-6"] == 4)]

dt = dt[dt[['asn']].apply(lambda x: x[0].isdigit(), axis=1)]
dt = dt[['asn', 'improvement_sc']]
print(dt.head())
print(dt.shape)
print(dt.dtypes)
print("------------------------------")

# needed to convert to a string first, then an float.
dt['asn'] = dt['asn'].astype(str).astype(float)
print(dt.dtypes)

# We need this format (LEFT Join), BUT we will use it later
# data = data.merge(dt, on="asn", how="left")
# print(data.head)
#
# data.to_csv('output.csv')

# We need left join, for now we will take inner
data = data.merge(dt, on="asn", how="inner")
print(data.head)

data.to_csv('output1.csv')

# Checking the missing values
print("-- Check for missing values --")
print(data.isnull().sum())
print("------------------------------")


list1 = [data.improvement_sc, data.asn, data.numberPrefixes, data.numberAsns, data.numberAddresses, data.total, data.customer, data.peer, data.provider]
for element in list1:
    print("------------------------------")
    print(element.name + " Analysis")
    print(element.describe())
    print("------------------------------")

#plot the scatter plot of AS_number and improvement_score variable in data
scatter_plot_list = [data.numberAsns, data.numberPrefixes, data.numberAddresses, data.total, data.customer, data.peer, data.provider]

for element in scatter_plot_list:
    plt.scatter(element, data.improvement_sc)
    plt.title("Scatter Plot")
    plt.xlabel(element.name)
    plt.ylabel("Improvement_score")
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()

# Correlation Matrix
data[['numberAsns', 'numberPrefixes', 'numberAddresses', 'total', 'customer', 'peer', 'provider', 'improvement_sc']].corr()
heatmap = sns.heatmap(data[['numberAsns', 'numberPrefixes', 'numberAddresses', 'total', 'customer', 'peer', 'provider', 'improvement_sc']].corr(), annot=True, cmap='Blues', cbar=False)
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig('Correlation Matrix.png')
plt.show()

# Plot ECDF
list_ECDF = [data.numberAsns, data.numberPrefixes, data.numberAddresses, data.total, data.customer, data.peer, data.provider]
for i in list_ECDF:
    my_ecdf = ECDF(i)
    plt.plot(my_ecdf.x, my_ecdf.y, marker='o')
    plt.xlabel(str(i.name))
    plt.ylabel('CDF')
    plt.title('Empirical CDF Plot')
    # plt.show()

# checking for each column the percentage of missing values
print("----------------------------")
for col in data.columns:
    pct_missing = np.mean(data[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing * 100)))
print("----------------------------")


y = data['improvement_sc']
X = data.drop(['improvement_sc', 'iso', 'asn', 'source', 'longitude', 'latitude'], axis=1)
print(X.columns)


# Train models without embeddings
x_train, x_test, y_train, y_test = split_data(X, y)
# Call models
models.train_models(X, x_train, x_test, y_train, y_test)


# SOS --> if we do not have --> header-None, dataframe deletes first row when we give name to the column
perso_data = pd.read_csv('../../Datasets/bgp.tools/perso.txt', header=None)
# name the column
perso_data.columns = ['asn']

# keep only the digits of the ASns
perso_data['asn'] = perso_data['asn'].apply(lambda x: keep_number(x))

# needed to convert to a string first, then an float.
perso_data['asn'] = perso_data['asn'].astype(str).astype(int)

# create a new column (True - False) if an asn exist in perso_data and data
data['matched'] = data['asn'].apply(lambda asn: exists_in_perso(perso_data, asn))
# True ==> 237
# print(data.matched[data.matched==True].count())

# Convert False to 0 and True to 1
data['matched'] = data['matched'].astype(int)
print(data.head())
print(data.shape)
print(data.dtypes)

# read embeddings (Node2Vec)
emb_df = pd.read_table("../../graph_embeddings/embeddings.emb", header=None, sep=" ")
# name the columns
rng = range(0, 65)
new_cols = ['dim_' + str(i) for i in rng]
emb_df.columns = new_cols
# rename first column
emb_df.rename(columns={'dim_0':'asn'}, inplace=True)
# emb_df.to_csv('aaa.csv')
print(emb_df.head())
print(emb_df.dtypes)

# data.drop_duplicates(subset=['asn'])
# emb_df.drop_duplicates(subset=['asn'])

print(emb_df.columns)
# emb_df.columns = emb_df.columns.astype(str).str.replace(' ', '')
data = data.merge(emb_df, on='asn', how="left", validate='many_to_many')
print(data.head())

data = data.fillna(0)
y = data['improvement_sc']
X = data.drop(['improvement_sc', 'iso', 'asn', 'source', 'longitude', 'latitude'], axis=1)

# # Train models with embeddings
# x_train, x_test, y_train, y_test = split_data(X, y)
# # Call models
# models.train_models(X, x_train, x_test, y_train, y_test)
