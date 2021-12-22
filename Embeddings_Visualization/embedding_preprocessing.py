import pandas as pd
import numpy as np
import pycountry_convert as pc
import pycountry
import os
from iso3166 import countries


PATH_AS_RELATIONSHIPS = '../Datasets/AS-relationships/20210701.as-rel2.txt'
NODE2VEC_EMBEDDINGS = '../Check_for_improvements/Embeddings/Node2Vec_embeddings.emb'
DEEPWALK_EMBEDDINGS_128 = '../Check_for_improvements/Embeddings/DeepWalk_128.csv'
DIFF2VEC_EMBEDDINGS_128 = '../Check_for_improvements/Embeddings/Diff2Vec_128.csv'
NETMF_EMBEDDINGS_128 = '../Check_for_improvements/Embeddings/NetMF_128.csv'
NODESKETCH_EMBEDDINGS_128 = '../Check_for_improvements/Embeddings/NodeSketch_128.csv'
WALKLETS_EMBEDDINGS_256 = '../Check_for_improvements/Embeddings/Walklets_256.csv'
NODE2VEC_EMBEDDINGS_64 = '../Check_for_improvements/Embeddings/Node2Vec_embeddings.emb'
NODE2VEC_LOCAL_EMBEDDINGS_64 = '../Check_for_improvements/Embeddings/Node2Vec_p2_64.csv'
NODE2VEC_GLOBAL_EMBEDDINGS_64 = '../Check_for_improvements/Embeddings/Node2Vec_q2_64.csv'
DIFF2VEC_EMBEDDINGS_64 = '../Check_for_improvements/Embeddings/Diff2Vec_64.csv'
NETMF_EMBEDDINGS_64 = '../Check_for_improvements/Embeddings/NetMF_64.csv'
NODESKETCH_EMBEDDINGS_64 = '../Check_for_improvements/Embeddings/NodeSketch_64.csv'
NODE2VEC_WL5_E3_LOCAL = '../Check_for_improvements/Embeddings/Node2Vec_64_wl5_ws2_ep3_local.csv'
NODE2VEC_WL5_E3_GLOBAL = '../Check_for_improvements/Embeddings/Node2Vec_64_wl5_ws2_ep3_global.csv'
NODE2VEC_64_WL5_E1_GLOBAL = '../Check_for_improvements/Embeddings/Node2Vec_64_wl5_ws2_global.csv'
BGP2VEC_64 = '../Check_for_improvements/Embeddings/Node2Vec_bgp2Vec.csv'
BGP2VEC_32 = '../Check_for_improvements/Embeddings/BGP2VEC_32'
WALKLETS_EMBEDDINGS_128 = '../Check_for_improvements/Embeddings/Walklets_128.csv'
STORE_CSV_TO_FOLDER = '../Embeddings_Visualization/StorePreprocessedEmb'


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
    data['AS_rank_iso'] = df['AS_rank_iso']

    return data['AS_rank_iso']


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


def get_path_and_filename(model, dimensions):
    """
    :param model: The model's name
    :param dimensions: The number of dimensions of the given model
    :return: The path where the script will be stored and its name
    """
    file_name = 'Preprocessed' + str(model) + str(dimensions) + f'.csv'
    outdir = STORE_CSV_TO_FOLDER
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    full_name = os.path.join(outdir, file_name)
    return full_name


def read_Node2Vec_embeddings_file():
    """
    :return: A dataframe containing the ASNs and the embeddings of each ASn created based on Node2Vec algorithm.
    """
    emb_df = pd.read_table(NODE2VEC_EMBEDDINGS, skiprows=1, header=None, sep=" ")
    # name the columns
    rng = range(0, 65)
    new_cols = ['dim_' + str(i) for i in rng]
    emb_df.columns = new_cols
    # rename first column
    emb_df.rename(columns={'dim_0': 'ASN'}, inplace=True)

    return emb_df


def read_karateClub_embeddings_file(emb, dimensions):
    """
    Karateclub library requires nodes to be named with consecutive Integer numbers. In the end gives as an output
    containing the embeddings in ascending order. So in this function we need to reassign each ASN to its own embedding.
    :param emb: A dataset containing pretrained embeddings
    :param dimensions: The dimensions of the given dataset
    :return: A dataframe containing pretrained embeddings
    """
    if dimensions == 64:
        if emb == 'Diff2Vec':
            df = pd.read_csv(DIFF2VEC_EMBEDDINGS_64, sep=',')
        elif emb == 'NetMF':
            df = pd.read_csv(NETMF_EMBEDDINGS_64, sep=',')
        elif emb == 'NodeSketch':
            df = pd.read_csv(NODESKETCH_EMBEDDINGS_64, sep=',')
        elif emb == 'Walklets':
            df = pd.read_csv(WALKLETS_EMBEDDINGS_128, sep=',')
        elif emb == 'Node2Vec_Local':
            df = pd.read_csv(NODE2VEC_LOCAL_EMBEDDINGS_64, sep=',')
        elif emb == 'Node2Vec_Global':
            df = pd.read_csv(NODE2VEC_GLOBAL_EMBEDDINGS_64, sep=',')
        elif emb == 'Node2Vec_wl5_global':
            df = pd.read_csv(NODE2VEC_64_WL5_E1_GLOBAL, sep=',')
        elif emb == 'Node2Vec_wl5_e3_global':
            df = pd.read_csv(NODE2VEC_WL5_E3_GLOBAL, sep=',')
        elif emb == 'Node2Vec_wl5_e3_local':
            df = pd.read_csv(NODE2VEC_WL5_E3_LOCAL, sep=',')
        elif emb == 'bgp2vec_64':
            df = pd.read_csv(BGP2VEC_64, sep=',')
        elif emb == 'bgp2vec_32':
            df = pd.read_csv(BGP2VEC_32, sep=',')
        else:
            raise Exception('Not defined dataset')
    else:
        if emb == 'Diff2Vec':
            df = pd.read_csv(DIFF2VEC_EMBEDDINGS_128, sep=',')
        elif emb == 'NetMF':
            df = pd.read_csv(NETMF_EMBEDDINGS_128, sep=',')
        elif emb == 'NodeSketch':
            df = pd.read_csv(NODESKETCH_EMBEDDINGS_128, sep=',')
        elif emb == 'Walklets':
            df = pd.read_csv(WALKLETS_EMBEDDINGS_256, sep=',')
        elif emb == 'DeepWalk':
            df = pd.read_csv(DEEPWALK_EMBEDDINGS_128, sep=',')
        else:
            raise Exception('Not defined dataset')

    df['0'] = df['0'].astype(int)
    if emb == 'Walklets':
        dimensions = dimensions * 2
    else:
        dimensions = dimensions
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