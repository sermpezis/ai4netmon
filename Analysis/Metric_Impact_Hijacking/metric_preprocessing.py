import pandas as pd
import numpy as np
import json
from iso3166 import countries
import pycountry_convert as pc
import pycountry
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.decomposition import PCA


ALL_RIPE_RIS_PEERS = '../../Datasets/RIPE_RIS_peers/list_of_RIPE_RIS_peers.json'
CAIDA_ASES = 'impact__CAIDA20190801_sims2000_hijackType0_per_monitor_onlyRC_NEW_with_mon_ASNs.csv'
BGP2VEC_32 = '../../Check_for_improvements/Embeddings/BGP2VEC_32'
PATH_AS_RELATIONSHIPS = '../../Datasets/AS-relationships/20210701.as-rel2.txt'


def read_ripe_ris_monitors():
    '''
    This function read RIPE RIS monitors and its only for testing purposes
    :return: The list of RIPE RIS monitors
    '''

    with open(ALL_RIPE_RIS_PEERS) as handle:
        dictdump = json.loads(handle.read())

    data = pd.DataFrame(dictdump.items(), columns=['IP_ADDRESS', 'ASN'])
    data.drop('IP_ADDRESS', axis=1, inplace=True)
    list_ripe = data.values.tolist()

    flat_list = [item for sublist in list_ripe for item in sublist]

    return flat_list


def compare_ases_from_caida_ripe(data_CAIDA, ripe_monitors):
    """
    This function read RIPE RIS monitors and its only for testing purposes
    :param data_CAIDA: A list containing ASes from CAIDA
    :param ripe_monitors: All ripe ris monitors
    :return: The number of common ASes from RIPE RIS and CAIDA
    """
    caida_Ases = data_CAIDA.columns.values.tolist()
    caida_Ases.pop()
    new_ases_caida = caida_Ases[7:]
    print(new_ases_caida)

    common_ases = []
    for i in new_ases_caida:
        for j in ripe_monitors:
            if i == str(j):
                common_ases.append(j)

    common_ases = list(dict.fromkeys(common_ases))
    print(len(common_ases))


def read_caida_ases():
    data_CAIDA = pd.read_csv(CAIDA_ASES, sep=",", header=0, dtype='unicode')

    # impact = label = #total hijacked ASes /  #total ASes with path to prefix
    data_CAIDA['impact'] = (data_CAIDA.iloc[:, 4].astype(float)) / (data_CAIDA.iloc[:, 2].astype(float))

    # delete rows where impact > 1 or impact < 0
    data_CAIDA = data_CAIDA.drop(data_CAIDA[(data_CAIDA.impact < 0) | (data_CAIDA.impact > 1)].index)

    # change the name of the column
    data_CAIDA.rename(columns={list(data_CAIDA)[2]: 'total_ASes_with_path_to_prefix'}, inplace=True)
    # delete rows where total ASes with path to prefix < 1000
    data_CAIDA = data_CAIDA.drop(data_CAIDA[(data_CAIDA.total_ASes_with_path_to_prefix.astype(float) < 1000.0)].index)

    return data_CAIDA


def one_hot(df):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """

    dummies = pd.get_dummies(df['AS_rank_iso'], prefix='Continent', drop_first=False)
    df = pd.concat([df, dummies], axis=1)
    return df


def country_flag(data):
    """
    :param data: Contains a dataframe combining 3 datasets
    :param list_alpha_2: Contains the 2-letter abbreviation from each country
    :return: Matches the acronyms with the Fullname of the countries
    """
    list_alpha_2 = [i.alpha2 for i in list(countries)]
    if data['AS_rank_iso'] in list_alpha_2:
        return pycountry.countries.get(alpha_2=data['AS_rank_iso']).name
    else:
        return 'Unknown Code'


def country_to_continent(country_name):
    """
    This function takes as input a country name and returns the continent that the given country belongs.
    :param country_name: Contains the name of a country
    :return: The continent
    """
    try:
        country_alpha2 = pc.country_name_to_country_alpha2(country_name)
        country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
        return country_continent_name
    except:
        return np.nan


def convert_country_to_continent(data):
    """
    The function converts iso = alpha_2 (example: US) to the whole name of the country. Needs (import iso3166)
    :param data: Contains a dataframe combining 4 datasets
    :return: The continent for each country
    """
    data['AS_rank_iso'] = data.apply(country_flag, axis=1)
    temp_list = []
    for i in range(0, len(data)):
        temp_list.append(country_to_continent(data['AS_rank_iso'][i]))
    df = pd.DataFrame(temp_list, columns=['AS_rank_iso'])
    data['AS_rank_iso'] = df

    return data['AS_rank_iso']


def read_karateClub_embeddings_file(dimensions):
    df = pd.read_csv(BGP2VEC_32, sep=',')
    df['0'] = df['0'].astype(int)

    rng = range(1, dimensions + 1)
    other_cols = ['dim_' + str(i) for i in rng]
    first_col = ['ASN']
    new_cols = np.concatenate((first_col, other_cols), axis=0)
    df.columns = new_cols

    # Replace the consecutive ASNs given from KarateClub library with the initial ASNs
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


def merge_datasets(final_df, embeddings_df):
    """
    :param final_df: Its the dataset that is generated in Analysis/aggregate_data folder
    :param embeddings_df: Contains pretrained embeddings
    :return: A new merged dataset (containing improvement_score and the embedding of each ASN)
    """
    print(final_df['ASN'].isin(embeddings_df['ASN']).value_counts())
    mergedStuff = pd.merge(embeddings_df, final_df, on=['ASN'], how='left')
    mergedStuff.replace('', np.nan, inplace=True)

    return mergedStuff