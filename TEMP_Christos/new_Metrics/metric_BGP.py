import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.decomposition import PCA
import call_models as cm

def one_hot(df):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """

    dummies = pd.get_dummies(data['iso'], prefix='Continent', drop_first=False)
    df = pd.concat([df, dummies], axis=1)
    return df

def caida_preprocess(data_CAIDA):
    # impact = label = #total hijacked ASes /  #total ASes with path to prefix
    data_CAIDA['impact'] = (data_CAIDA.iloc[:, 4].astype(float)) / (data_CAIDA.iloc[:, 2].astype(float))

    # delete rows where impact > 1 or impact < 0
    data_CAIDA = data_CAIDA.drop(data_CAIDA[(data_CAIDA.impact < 0) | (data_CAIDA.impact > 1)].index)

    # change the name of the column
    data_CAIDA.rename(columns={list(data_CAIDA)[2]: 'total_ASes_with_path_to_prefix'}, inplace=True)
    # delete rows where total ASes with path to prefix < 1000
    data_CAIDA = data_CAIDA.drop(data_CAIDA[(data_CAIDA.total_ASes_with_path_to_prefix.astype(float) < 1000.0)].index)
    return data_CAIDA

data_CAIDA = pd.read_csv('impact__CAIDA20190801_sims2000_hijackType0_per_monitor_onlyRC_NEW_with_mon_ASNs.csv', sep=",", dtype='unicode')

new_data_CAIDA = caida_preprocess(data_CAIDA)

data = pd.read_csv('../AS_improvement_scores/metric_data.csv', sep=",", dtype='unicode')
# drop the first and second column
cols = [0, 1]
data = data.drop(data.columns[cols], axis=1)

# drop: source, improvement_sc, longitude, latitude
data = data.drop(['source', 'improvement_sc', 'longitude', 'latitude', 'top_k'], axis=1)
# keep AS number as index
data = data.set_index('asn')

# Iso has 7 unique values, so the one hot encoding will give us 7 columns
print(data.iso.nunique())
# with embeddings
data_with_emb = one_hot(data)
data_with_emb = data_with_emb.drop(['iso'], axis=1)

# without embeddings
rng = range(1, 65)
new_cols = ['dim_' + str(i) for i in rng]
data_without_emb = data_with_emb.drop(new_cols, axis=1)

new_data_CAIDA = new_data_CAIDA.fillna(0)
print(data_without_emb.columns)

# random selection of 50 monitors (columns)
final_data = []
for index, row in new_data_CAIDA.iterrows():

    # keep the impact of each row
    new_row = str(row['impact'])
    # select only columns that belong to a monitor --> observations and belong to a specific row
    # rand_monitors is a list which contains 50 randomly selected monitors
    # and I should keep the name of the column with the sample
    randomlist = random.sample(range(7, 287), 50)
    # values from 50 random columns
    rand_monitors = row.iloc[randomlist]
    # keep from the above 50 values column's name
    columns_names = new_data_CAIDA.columns[randomlist]

    rand_monitors = np.array(rand_monitors, dtype=np.str)
    new_row = [new_row] + [x for x in rand_monitors]

    # for each monitor I want to keep the features from the dataframe --> data_without_emb
    for name in columns_names:
        if name in data_without_emb.index:
            test = np.array(data_without_emb.loc[name], dtype=np.str)
            new_row = np.concatenate((new_row, test), axis=0)
        else:
            # We need to fill in the 16 columns of data_without_emb with zero + 1 column observation_i
            listofzeros = [0 for i in range(15)]
            new_row = np.concatenate((new_row, listofzeros), axis=0)
    final_data.append(new_row.tolist())
# print(final_data)

first_col = ['impact', 'observation_1']
# I need 49 more columns like the col1 but without impact and observation + 1 each time
obs_rng = range(2, 51)
middle_cols = ['observation_' + str(i) for i in obs_rng]
other_cols = ['rank', 'numberAsns', 'numberPrefixes', 'numberAddresses', 'total', 'customer', 'peer', 'provider', 'Continent_Africa',
              'Continent_Asia', 'Continent_Europe', 'Continent_No info', 'Continent_North America', 'Continent_Oceania', 'Continent_South America']

# union all column names to one list
counter = 1
for i in middle_cols:
    first_col.append(i)

for i in range(1, 51):
    counter += 1
    for j in other_cols:
        first_col.append(j + str(counter))

final_df = pd.DataFrame(final_data, columns=first_col)

cols = [i for i in final_df.columns if i not in ["Impact"]]
for col in cols:
    final_df[col] = pd.to_numeric(final_df[col])
# print(final_df.dtypes)

final_df.to_csv("Impact_&_features.csv")
y = final_df['impact']
X = final_df.drop(['impact'], axis=1)
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20)

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

method = 'PCA'

if method == 'PCA':
    # apply PCA keeping a certain number of features
    # n_components --> 1<= max value <= 397
    pca = PCA(n_components=397)
    pca_x_train = pca.fit_transform(x_train)
    pca_x_test = pca.fit_transform(x_test)
    print(pca_x_train.shape)
    print(pca_x_test.shape)
    cm.ml_models(pca_x_train, pca_x_test, y_train, y_test)
elif method == 'Cross_Validation':
    cm.cross_val_with_Random_Forest(final_df, X, y)
else:
    cm.ml_models(x_train, x_test, y_train, y_test)

