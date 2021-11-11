import pandas as pd
import numpy as np

DIFF2VEC_EMBEDDINGS = 'Embeddings/Diff2Vec_embeddings.csv'
NETMF_EMBEDDINGS = 'Embeddings/NetMF_embeddings.csv'

PATH_AS_RELATIONSHIPS = '../Datasets/AS-relationships/20210701.as-rel2.txt'
data = pd.read_csv(PATH_AS_RELATIONSHIPS, sep="|", skiprows=180, header=None)
data.columns = ['source', 'target', 'link', 'protocol']
data.drop(['link', 'protocol'], axis=1, inplace=True)

unique_nodes1 = set(data.source)
unique_nodes2 = set(data.target)

all_nodes = set(unique_nodes1.union(unique_nodes2))
sort_nodes = sorted(all_nodes)

previous_data = pd.DataFrame(sort_nodes)


df = pd.read_csv(DIFF2VEC_EMBEDDINGS, sep=',')
df['0'] = df['0'].astype(int)


dimensions = 64
rng = range(1, dimensions + 1)
other_cols = ['dim_' + str(i) for i in rng]
first_col = ['ASN']
new_cols = np.concatenate((first_col, other_cols), axis=0)
df.columns = new_cols


final_df = pd.concat([previous_data, df], axis=1)
final_df.drop('ASN', axis=1, inplace=True)
final_df.rename({0: 'ASN'}, axis=1, inplace=True)
# final_df.to_csv('aaaaa.csv', index=False)
print(final_df)
