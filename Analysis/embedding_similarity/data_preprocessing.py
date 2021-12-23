import pandas as pd
import numpy as np
import pycountry_convert as pc
import pycountry
from iso3166 import countries


def merge_datasets(final_df, embeddings_df):
    """
    :param final_df: Its the dataset that is generated in Analysis/aggregate_data folder
    :param embeddings_df: Contains the similarity matrix
    :return: A new merged dataset (containing improvement_score and the embedding of each ASN)
    """
    print(final_df['ASN'].isin(embeddings_df['ASN']).value_counts())
    mergedStuff = pd.merge(embeddings_df, final_df, on=['ASN'], how='left')
    mergedStuff.replace('', np.nan, inplace=True)

    return mergedStuff


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
