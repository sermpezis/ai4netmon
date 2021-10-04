import pandas as pd
import networkx as nx
import re
import json


def keep_number(text):

    """
    :param text: example AS206924
    :return: 206924
    """
    num = re.findall(r'[0-9]+', text)

    return " ".join(num)

def create_df_from_AS_rank():

    """
   Change the column names in order to know the features origin
   :return: return the new dataframe
   """
    path = '../../Datasets/As-rank/asns.csv'
    data = pd.read_csv(path, sep=",")
    new_columns = ['AS_rank_' + str(i) for i in data.columns]
    data = data.set_axis(new_columns, axis='columns', inplace=False)
    data = data.set_index('AS_rank_asn')

    return data

def create_df_from_personal():

    """
    :return: the a dataframe which contains only one column. This column has the ASn of personal dataset as integers
    """
    path = '../../Datasets/bgp.tools/perso.txt'
    data = pd.read_csv(path, header=None)
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
    path = '../../Datasets/PeeringDB/peeringdb_2_dump_2021_07_01.json'
    df = pd.read_json(path)
    data = []
    keep_keys = ['info_ratio', 'info_traffic', 'info_scope', 'info_type', 'info_prefixes4',
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
    data = df

    return data

def create_df_from(dataset):

    if dataset =='AS_rank':

       data = create_df_from_AS_rank()
       return data

    elif dataset == 'personal':

        data = create_df_from_personal()
        return data

    elif dataset == 'PeeringDB':

        data = create_df_from_PeeringDB()
        return data

def dfs_concatanate(data_AS, data_per):

    """
    :param data_AS: dataframe AS_rank
    :param data_per: dataframe AS_personal
    :return: the concatenation of the 2 dataframes
    """

    data = pd.concat([data_AS, data_per], axis=1)
    return data

def create_your_dataframe():

    """
    This function creates the user's requested dataframe
    :return: The requested dataframe
    """
    # Create an empty DataFrame object
    data = pd.DataFrame()

    list_od_datasets = ['AS_rank', 'personal', 'PeeringDB']
    list_of_dataframes = []
    for i in list_od_datasets:
        list_of_dataframes.append(create_df_from(i))
    data_con = dfs_concatanate(list_of_dataframes[0], list_of_dataframes[1])

    return data_con

def create_bigraph_from_AS_relationships():

    """
    This function takes as input 20210701.as-rel2.txt  and creates a graph based on NetworkX library.
    :return: A bipartite graph containing only peers nodes
    """
    path = '../../Datasets/AS-relationships/20210701.as-rel2.txt'
    data = pd.read_csv(path, sep="|", skiprows=180)
    data.columns = ['node1', 'node2', 'link', 'protocol']
    data.drop(['protocol'], axis=1, inplace=True)


    graph = nx.Graph()
    graph.add_nodes_from(data.node1, bipartite=0)
    graph.add_nodes_from(data.node2, bipartite=1)

    for node1, node2, link in data.values:
        # Check if node1 and node2 are peers
        if link == 0:
            graph.add_edge(node1, node2)

    return graph

def create_bigraph_from_netixlan():

    """
    This function takes as input netixlan.json  and creates a graph based on NetworkX library.
    :return: A bipartite graph ( with node1=ixlan_id and node2=asn)
    """
    path = '../../Datasets/PeeringDB/netixlan.json'
    data = json.load(open(path))
    df = pd.DataFrame(data["data"])
    df.drop(['id', 'net_id', 'ix_id', 'name', 'notes', 'speed', 'ipaddr4', 'ipaddr6', 'is_rs_peer', 'operational', 'created', 'updated', 'status'], axis=1, inplace=True)

    graph = nx.Graph()
    graph.add_nodes_from(df.ixlan_id, bipartite=0)
    graph.add_nodes_from(df.asn, bipartite=1)

    for node1, node2 in df.values:
        graph.add_edge(node1, node2)

    return graph
