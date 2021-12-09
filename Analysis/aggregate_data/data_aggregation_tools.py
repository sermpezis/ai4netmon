import pandas as pd
import numpy as np
import networkx as nx
import json
import pycountry_convert as pc

PATH_AS_RANK = '../../Datasets/AS-rank/asns.csv'
PATH_PERSONAL = '../../Datasets/bgp.tools/perso.txt'
PATH_PEERINGDB = '../../Datasets/PeeringDB/peeringdb_2_dump_2021_07_01.json'
PATH_AS_RELATIONSHIPS = '../../Datasets/AS-relationships/20210701.as-rel2.txt'
PATH_PEERINGDB_NETIXLAN = '../../Datasets/PeeringDB/netixlan.json'
PATH_BGP = '../bgp_paths/random_data.txt'
AS_HEGEMONY_PATH = '../../Datasets/AS-hegemony/AS_hegemony.csv'
ALL_ATLAS_PROBES = '../../Datasets/RIPE_Atlas_probes/bq_results.json'


def cc2cont(country_code):
    '''
    Receives a country code ISO2 (e.g., 'US') and returns the corresponding continent name (e.g., 'North America'). 
    Exceptions: 
        - if 'EU' is given as country code (it happened in data), then it is treated as the continent code
        - if the country code is not found, then a None value is returned
    :param  country_code:   (str) ISO2 country code
    :return:    (str) continent name of the given country(-ies)
    '''
    if country_code in ['EU']:
        continent_code = country_code
    else:
        try:
            continent_code = pc.country_alpha2_to_continent_code(country_code)
        except KeyError:
            return None
    continent_name = pc.convert_continent_code_to_continent_name(continent_code)
    return continent_name
    

def get_continent(country_code):
    '''
    Receives a series of country codes ISO2 (e.g., 'US') and returns the corresponding continent names (e.g., 'North America'). 
    For NaN or None elements, it returns a None value
    :param  country_code:   (pandas Series) ISO2 country codes
    :return:    (list of str) continent names of the given countries
    '''
    continent_name = []
    for cc in country_code.tolist():
        if pd.isna(cc):
            continent_name.append(None)
        else:
            continent_name.append( cc2cont(cc) )
    return continent_name


def create_df_from_Atlas_probes():
    """
    Loads the list of RIPE Atlas probes, and returns a dataframe with the number of v4 and v6 probes per ASN (only for ASNs that have at least one probe).
    :return: A dataframe with index the ASN
    """
    data = pd.read_json(ALL_ATLAS_PROBES, lines=True)
    data = data[(data['status'] == 'Connected')]
    s4 = data['asn_v4'].value_counts()
    s6 = data['asn_v6'].value_counts()
    df = pd.concat([s4, s6], axis=1)
    df.index.name = 'asn'
    df = df.rename(columns={'asn_v4': 'nb_atlas_probes_v4', 'asn_v6': 'nb_atlas_probes_v6'})

    return df


def create_df_from_AS_rank():
    """
    Loads the CAIDA AS-rank dataset from the source file. Returns a dataframe with index the ASN; appends in the column names the prefix "AS_rank_".
    :return: A dataframe with index the ASN
    """
    data = pd.read_csv(PATH_AS_RANK, sep=",")
    new_columns = ['AS_rank_' + str(i) for i in data.columns]
    data = data.set_axis(new_columns, axis='columns', inplace=False)
    data.loc[(data['AS_rank_longitude'] == 0) & (data['AS_rank_latitude'] == 0), ['AS_rank_longitude',
                                                                                  'AS_rank_latitude']] = None
    data['AS_rank_continent'] = get_continent(data['AS_rank_iso'])
    data = data.set_index('AS_rank_asn')

    return data


def create_df_from_AS_hegemony():
    """
    Loads the AS hegemony dataset from the source file. Returns a dataframe with index the ASN, and a single column with the AS hegemony value of the AS
    :return: A dataframe with index the ASN
    """
    data = pd.read_csv(AS_HEGEMONY_PATH, sep=",")
    data = data.rename(columns={'hege': 'AS_hegemony'})
    data = data.set_index('asn')

    return data


def create_df_from_personal():
    """
    Loads the bgp.tools personal AS dataset from the source file. Creates a dataframe with index the ASN of the ASes that are personal use ASes; the dataframe has only one column with 1 for all rows
    :return: A dataframe with index the ASN
    :return: the a dataframe which contains only one column.
    """
    data = pd.read_csv(PATH_PERSONAL, header=None)
    data.columns = ['asn']
    # keep only the digits of the ASNs
    data['asn'] = data['asn'].apply(lambda x: int(x[2:]))
    data['is_personal_AS'] = 1
    data = data.set_index('asn')

    return data


def create_df_from_PeeringDB():
    """
    Loads the PeeringDB dataset from the source file. Returns a dataframe with index the ASN; appends in the column names the prefix "peeringDB_". The dataframe which contains only the features in the keep_keys list
    :return: A dataframe with index the ASN
    """
    df = pd.read_json(PATH_PEERINGDB)
    data = []
    keep_keys = ['asn', 'info_ratio', 'info_traffic', 'info_scope', 'info_type', 'info_prefixes4',
                 'info_prefixes6', 'policy_general', 'ix_count', 'fac_count', 'created']
    for row in df.net['data']:
        net_row = [row.get(key) for key in keep_keys]
        data.append(net_row)
    df = pd.DataFrame(data, columns=keep_keys)
    new_columns = ['peeringDB_' + str(i) for i in df.columns]
    df = df.set_axis(new_columns, axis='columns', inplace=False)
    df = df.set_index('peeringDB_asn')

    return df


def create_df_from(dataset):
    """
    Selects a method, based on the given dataset name, and creates the corresponding dataframe.
    When adding a new method, take care to have as index the ASN and the column names to be of the format "dataset_name_"+"column_name" (e.g., the column "X" from the dataset "setA", should be "setA_X") 
    :param dataset: (type = string) name of the dataset to be loaded
    :return: A dataframe with indexes the ASNs and columns the features loaded from the given dataset
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
        raise Exception('Not defined dataset')
    return data


def create_dataframe_from_multiple_datasets(list_of_datasets):
    """
    Creates a dataframe for each given dataset, and concatenates all the dataframes in a common dataframe. The final/returned dataframe has the ASN as the index, and as columns all the columns from all datasets. It fills with NaN non existing values. 
    :param list_of_datasets:    a list of str, where each string corresponds to a dataset to be loaded 
    :return: A dataframe with indexes the ASNs and columns the features loaded from each given dataset
    """
    data = pd.DataFrame()
    list_of_dataframes = []
    for i in list_of_datasets:
        list_of_dataframes.append(create_df_from(i))
    final_df = pd.concat(list_of_dataframes, axis=1)
    final_df.index.name = 'ASN'
    return final_df


def create_graph_from_AS_relationships():
    """
    This function takes as input 20210701.as-rel2.txt  and creates a graph based on NetworkX library.
    :return: A graph
    """

    data = pd.read_csv(PATH_AS_RELATIONSHIPS, sep="|", skiprows=180, header=None)
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
