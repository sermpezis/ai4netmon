import pandas as pd
import numpy as np
import networkx as nx
import numpy as np
import re
import json

PATH_AS_RANK = '../../Datasets/As-rank/asns.csv'
PATH_PERSONAL = '../../Datasets/bgp.tools/perso.txt'
PATH_PEERINGDB = '../../Datasets/PeeringDB/peeringdb_2_dump_2021_07_01.json'
PATH_AS_RELATIONSHIPS = '../../Datasets/AS-relationships/20210701.as-rel2.txt'
PATH_PEERINGDB_NETIXLAN = '../../Datasets/PeeringDB/netixlan.json'
PATH_BGP = '../bgp_paths/random_data.txt'
AS_HEGEMONY_PATH = '../../Datasets/AS_hegemony/AS_hegemony.csv'
ALL_ATLAS_PROBES = '../../Datasets/Atlas_probe/bq_results.json'

def keep_number(text):
    """
    :param text: example AS206924
    :return: 206924
    """
    num = re.findall(r'[0-9]+', text)

    return " ".join(num)

def create_df_from_Atlas_probes():
    """
    :return: The new dataframe where when 2 or more probes have the same AS number (only for asn_v4) returned in the same row
    """

    data = pd.read_json(ALL_ATLAS_PROBES, lines=True)
    data.drop_duplicates(subset="prb_id", keep='first', inplace=True)
    data = data[(data['status'] == 'Connected')]


    data = data.assign(asn=pd.Series(np.random.randn(len(data['status']))).values)
    asn_list = []
    for i in range(0, len(data)):
        if data['asn_v4'].iloc[i] == np.nan:
            asn_list.append([data['prb_id'].iloc[i], data['asn_v4'].iloc[i], data['asn_v6'].iloc[i], data['status'].iloc[i], data['asn_v6'].iloc[i]])
        elif data['asn_v6'].iloc[i] == np.nan:
            asn_list.append([data['prb_id'].iloc[i], data['asn_v4'].iloc[i], data['asn_v6'].iloc[i], data['status'].iloc[i], data['asn_v4'].iloc[i]])
        elif data['asn_v4'].iloc[i] == data['asn_v6'].iloc[i]:
            asn_list.append([data['prb_id'].iloc[i], data['asn_v4'].iloc[i], data['asn_v6'].iloc[i], data['status'].iloc[i], data['asn_v4'].iloc[i]])
        elif data['asn_v4'].iloc[i] != data['asn_v6'].iloc[i]:
            asn_list.append([data['prb_id'].iloc[i], data['asn_v4'].iloc[i], data['asn_v6'].iloc[i], data['status'].iloc[i], data['asn_v4'].iloc[i]])
            asn_list.append([data['prb_id'].iloc[i], data['asn_v4'].iloc[i], data['asn_v6'].iloc[i], data['status'].iloc[i], data['asn_v6'].iloc[i]])

    df = pd.DataFrame(asn_list, columns=['prb_id', 'asn_v4', 'asn_v6', 'status', 'asn'])
    df.dropna()
    df.drop_duplicates()
    df['status'] = (df['status'] == 'Connected').astype(int)
    df.rename(columns={'status': 'has_atlas_probe'}, inplace=True)
    df = df.groupby(['asn', 'has_atlas_probe'])['prb_id'].apply(lambda x: ','.join(x.astype(str))).reset_index()
    df = df.set_index('asn')

    return df


def create_df_from_AS_rank():
    """
   Change the column names in order to know the features origin
   :return: return the new dataframe
   """
    data = pd.read_csv(PATH_AS_RANK, sep=",")
    new_columns = ['AS_rank_' + str(i) for i in data.columns]
    data = data.set_axis(new_columns, axis='columns', inplace=False)
    data = data.set_index('AS_rank_asn')

    return data


def create_df_from_AS_hegemony():
    """
   :return: return the new dataframe with Asn column as index
   """
    data = pd.read_csv(AS_HEGEMONY_PATH, sep=",")
    data = data.set_index('asn')

    return data


def create_df_from_personal():
    """
    :return: the a dataframe which contains only one column. This column has the ASn of personal dataset as integers
    """

    data = pd.read_csv(PATH_PERSONAL, header=None)
    # name the column
    data.columns = ['asn']
    # keep only the digits of the ASns
    data['asn'] = data['asn'].apply(lambda x: keep_number(x))
    data['personal_is_matched'] = 1
    # needed to convert to a string first, then to an int.
    data['asn'] = data['asn'].astype(str).astype(int)
    data = data.set_index('asn')

    return data


def create_df_from_PeeringDB():
    """
    :return PeeringDB dataframe which contains only the features in the keep_keys list.
    """
    df = pd.read_json(PATH_PEERINGDB)
    data = []
    keep_keys = ['asn', 'info_ratio', 'info_traffic', 'info_scope', 'info_type', 'info_prefixes4',
                 'info_prefixes6', 'policy_general', 'ix_count', 'fac_count', 'created']
    for row in df.net['data']:
        net_row = []
        for key in keep_keys:
            if key in row:
                net_row.append(row[key])
            else:
                net_row.append(None)
        data.append(net_row)
    df = pd.DataFrame(data, columns=keep_keys)
    # rename column names add the prefix peeringDB_
    new_columns = ['peeringDB_' + str(i) for i in df.columns]
    df = df.set_axis(new_columns, axis='columns', inplace=False)
    df = df.set_index('peeringDB_asn')
    data = df

    return data


def check_if_concatenate_works_properly(list_of_dataframes):
    """
    :param list_of_dataframes: It contains all the datasets in a dataframe form
    :return: the number of rows that our csv should have, after the correct concatenation of the datasets
    """
    union_indices = []
    for df in list_of_dataframes:
        idx = df.index
        union_indices = np.union1d(union_indices, idx)
    idx = len(union_indices)
    # print(idx)


def create_df_from(dataset):
    """
    In case user give error names for our dataset we print him an Exception and the program is finished
    :param dataset: (type = string) Accepted parameters: The name should exist in the datasets
    :return: A dataframe that has as index the ASn feature
    """
    if dataset == 'AS_rank':
        data = create_df_from_AS_rank()
    elif dataset == 'personal':
        data = create_df_from_personal()
    elif dataset == 'PeeringDB':
        data = create_df_from_PeeringDB()
    elif dataset == 'AS_hegemony':
        data = create_df_from_AS_hegemony()
    elif dataset == 'Atlas_probes':
        data = create_df_from_Atlas_probes()
    else:
        raise Exception('Not defined type of dataset')
    return data


def create_dataframe_from_multiple_datasets(list_of_datasets):
    """
    This function concatenates all the given datasets
    :return: The requested dataframe
    """
    # Create an empty DataFrame object
    data = pd.DataFrame()
    list_of_dataframes = []
    for i in list_of_datasets:
        list_of_dataframes.append(create_df_from(i))

    check_if_concatenate_works_properly(list_of_dataframes)

    return pd.concat(list_of_dataframes, axis=1, ignore_index=False).reindex(list_of_dataframes[0].index)


def create_graph_from_AS_relationships():
    """
    This function takes as input 20210701.as-rel2.txt  and creates a graph based on NetworkX library.
    :return: A graph
    """

    data = pd.read_csv(PATH_AS_RELATIONSHIPS, sep="|", skiprows=180)
    data.columns = ['node1', 'node2', 'link', 'protocol']
    data.drop(['protocol'], axis=1, inplace=True)

    graph = nx.Graph()
    graph.add_nodes_from(data.node1, node_type="node")
    graph.add_nodes_from(data.node2, node_type="node")

    for line in data.values:
        graph.add_edge(line[0], line[1])

    return graph


def create_bigraph_from_netixlan():
    """
    This function takes as input netixlan.json  and creates a graph based on NetworkX library.
    :return: A bipartite graph (with node1=ixlan_id and node2=asn)
    """

    data = json.load(open(PATH_PEERINGDB_NETIXLAN))
    df = pd.DataFrame(data["data"])
    df = df[['ixlan_id', 'asn']]

    graph = nx.Graph()
    graph.add_nodes_from(df.ixlan_id, bipartite=0)
    graph.add_nodes_from(df.asn, bipartite=1)

    for node1, node2 in df.values:
        graph.add_edge(node1, node2)

    return graph


def create_graph_from_bgp_paths(paths):
    """
    :param paths: A list of lists containing paths (random_generated)
    :return: A graph based on the random generated paths
    """
    columns = ['node' + str(i) for i in range(1, 16)]
    df = pd.DataFrame(columns=columns, data=paths)
    print(df)
    graph = nx.Graph()

    for node in df:
        for i in range(0, len(df)):
            graph.add_nodes_from([df[node].values[i]], node_type="node")

    for i in range(0, len(df)):
        for j in range(1, len(df.columns)):
            graph.add_edges_from([(df['node{}'.format(j)].values[i], df['node{}'.format(j + 1)].values[i])])

    return graph
