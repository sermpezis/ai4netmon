import pandas as pd
import numpy as np
import random
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
data = data.drop(['source', 'improvement_sc', 'longitude', 'latitude'], axis=1)
# keep AS number as index
data = data.set_index('asn')
print(data.head())

# with embeddings
data_with_emb = one_hot(data)


# without embeddings
rng = range(1, 65)
new_cols = ['dim_' + str(i) for i in rng]
data_without_emb = data_with_emb.drop(new_cols, axis=1)
data_without_emb = one_hot(data)
print(data_without_emb.head())

new_data_CAIDA = new_data_CAIDA.fillna(0)

# random selection of 50 monitors (columns)
for index, row in new_data_CAIDA.iterrows():

    # keep the impact of each row
    new_row = str(row['impact'])
    # select only columns that belong to a monitor --> observations and belong to a specific row
    # rand_monitors is a list which contains 50 randomly selected monitors
    # and I should keep the name of the column with the sample
    randomlist = random.sample(range(7, 287), 50)
    # rand_monitors = row.iloc[7:287].sample(n=50, axis=0)
    # values from 50 random columns
    rand_monitors = row.iloc[randomlist]
    # keep from the above 50 values column's name
    columns_names = new_data_CAIDA.columns[randomlist]

    rand_monitors = np.array(rand_monitors, dtype=np.str)
    new_row = [new_row] + [x for x in rand_monitors]
    # print(new_row)

    # for each monitor I want to keep the features from the dataframe --> data_without_emb
    for name in columns_names:
        if name in data_without_emb.index:
            test = np.array(data_without_emb.loc[name], dtype=np.str)
            new_row = np.concatenate((new_row, test), axis=0)
            print(new_row)
#pd.DataFrame(new_row).to_csv("foo1.csv")
