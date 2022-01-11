import pandas as pd
import numpy as np
import json

EMBEDDING_PATH = '../../../Check_for_improvements/Embeddings/BGP2Vec_32_wl6_ws5_ep3_wn40_p4_q05.csv'
PATH_AS_RELATIONSHIPS = '../../../Datasets/AS-relationships/20210701.as-rel2.txt'
RIPE_RIS_PEERS = '../../../Datasets/RIPE_RIS_peers/improvements_RIPE_RIS_peers_leave_one_out.json'
ALL_RIPE_RIS_PEERS = '../../../Datasets/RIPE_RIS_peers/list_of_RIPE_RIS_peers.json'
all_ripe_peers = True
dimensions = 32


def read_dataset():
    """
    Karateclub library requires nodes to be named with consecutive Integer numbers. After running the library, we take
    as output the embeddings in ascending order. So in this function, we need to reassign each ASN to its own embedding.
    :return: A dataframe containing tha AS number and its embedding
    """
    df = pd.read_csv(EMBEDDING_PATH, sep=',')
    rng = range(1, dimensions + 1)
    other_cols = ['dim_' + str(i) for i in rng]
    first_col = ['ASN']
    new_cols = np.concatenate((first_col, other_cols), axis=0)
    df.columns = new_cols

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


def read_RIS_improvement_score(all_peers):
    """
    :return: A dataframe containing 2 columns. The first one contains monitors and the second one the improvement score
     that each monitor will bring to the Network
    """
    if all_peers:
        with open(ALL_RIPE_RIS_PEERS) as handle:
            dictdump = json.loads(handle.read())

        data = pd.DataFrame(dictdump.items(), columns=['IP_ADDRESS', 'ASN'])
        data['ASN'] = data['ASN'].astype(np.int64)
    else:
        with open(RIPE_RIS_PEERS) as handle:
            dictdump = json.loads(handle.read())

        data = pd.DataFrame(dictdump.items(), columns=['ASN', 'Improvement_score'])

        data['ASN'] = data['ASN'].astype(np.int64)
    return data


def merge_datasets(improvement_df, embeddings_df):
    """
    :param improvement_df: Contains the improvement score that each monitor will bring to the network
    :param embeddings_df: Contains pretrained embeddings
    :return: A new merged dataset (containing improvement_score and the embedding of each monitor)
    """
    print(improvement_df['ASN'].isin(embeddings_df['ASN']).value_counts())
    mergedStuff = pd.merge(improvement_df, embeddings_df, on=['ASN'], how='left')
    mergedStuff['ASN'] = mergedStuff['ASN'].astype(float)
    mergedStuff.replace('', np.nan, inplace=True)

    return mergedStuff


def calculate_euclidean_distance(new_data):
    """
    :param new_data: Contains the graph embeddings for each RIPE RIS monitor
    :return: The column names for the dataframe and a list of lists containing the distance matrix
    """
    # store the ASNs in a list
    monitors = ['IP_ADDRESS'] + new_data['IP_ADDRESS'].tolist()
    list = []
    final_list = [[] for i in range(len(new_data.index))]
    for i in range(len(new_data.index)):
        for j in range(len(new_data.index)):
            list.append(np.linalg.norm(new_data.iloc[i, 1:] - new_data.iloc[j, 1:]))
            if j >= len(new_data) - 1:
                final_list[i].append([new_data.iloc[i, 0]] + list)
                list = []
    # covert a list of lists of lists to a list of lists
    flat_list = [item for sublist in final_list for item in sublist]

    return monitors, flat_list


improvement_df = read_RIS_improvement_score(all_ripe_peers)
embeddings_df = read_dataset()
data = merge_datasets(improvement_df, embeddings_df)
if all_ripe_peers:
    data.drop('ASN', axis=1, inplace=True)
else:
    data.drop('Improvement_score', axis=1, inplace=True)
print(data)
monitors, flat_list = calculate_euclidean_distance(data)
df = pd.DataFrame(flat_list, columns=monitors)
df.to_csv('ALL_RIPE_RIS_distance_embeddings_BGP2VEC_20210107.csv', sep=',', index=False)