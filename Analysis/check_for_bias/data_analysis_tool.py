import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from iso3166 import countries
import pycountry_convert as pc
import pycountry
import re
from datetime import datetime
from statsmodels.distributions.empirical_distribution import ECDF


FINAL_DATAFRAME = '../aggregate_data/final_dataframe.csv'
PATH_RIPE_RIS_PEERS = '../../Datasets/RIPE_RIS_peers_monitors/list_of_RIPE_RIS_peers.json'
RIPE_ATLAS_PROBES = '../../Datasets/Atlas_probe/bq_results.json'


def read_ripe_peers():
    """
    :return: A dataframe with one column that contains all RIPE peer-monitors ASn
    """
    data_ripe = pd.read_json(PATH_RIPE_RIS_PEERS, typ='dictionary')
    list_of_ripe_peers = [i for i in data_ripe.values]
    list_of_uniques_ripe_peers = set(list_of_ripe_peers)
    # convert list to dataframe
    df = pd.DataFrame(list_of_uniques_ripe_peers, columns=['ASn'])

    return df


def take_unique_ATLAS_ASNs():
    """
    :return: The returned set contains only the unique ASns
    """

    data = pd.read_json(RIPE_ATLAS_PROBES, lines=True)
    data = data[(data['status'] == 'Connected')]
    set1 = set(data['asn_v4'])
    set2 = set(data['asn_v6'])
    union = set.union(set1, set2)
    union = {x for x in union if pd.notna(x)}

    atlas_dataframe = pd.DataFrame(union)
    atlas_dataframe.columns = ['Atlas_ASN']

    return atlas_dataframe


def read_final_dataframe():
    """
    :return: A dataframe that is created from the concatenation of 5 datasets
    """

    return pd.read_csv(FINAL_DATAFRAME, dtype={"prb_id": "string"}, sep=',')


def call_categorize(final_df, current_df, type):
    """
    :param final_df: A dataframe that is created from the concatenation of 5 datasets
    :param current_df: Contains the unique AS numbers of RIPE RIS or the AS numbers of RIPE ATLAS probes
    :param type: Can take 2 values ('Ripe_Ris_monitors', 'Ripe_Atlas_probes')
    """

    for column_name in final_df.columns:
        dataTypeObj = final_df.dtypes[column_name]
        categorize_features(final_df, current_df, dataTypeObj, column_name, type)


def convert_to_numerical(data):
    """
    The function subtracts the created year of peeringDB from the current year.
    :param data: It contains all features from 3 different datasets
    :return: A numerical feature containing the above described subtraction
    """
    data['peeringDB_created'] = data['peeringDB_created'].astype('str')
    data['peeringDB_created'] = data['peeringDB_created'].apply(lambda x: keep_number(x))
    today_year = datetime.today()
    data['peeringDB_created'] = data['peeringDB_created'].apply(lambda x: int(today_year.year) - int(x))

    return data['peeringDB_created']

def keep_number(text):
    """
    :param text: example 2005-06-10T02:28:32Z
    :return: Only the year --> 2005
    """
    if text == '0':
        return '0000'
    else:
        num = re.findall(r'\d{4}', text)
        return "".join(num)


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


def categorize_features(data, current, type, feature, type_of_monitors):
    if type == np.int64 or type == np.float64:
        if feature == 'peeringDB_info_prefixes4':
            data['peeringDB_info_prefixes4'] = data.peeringDB_info_prefixes4.fillna(0)
            data['peeringDB_info_prefixes4'] = data.peeringDB_info_prefixes4.astype('Int64')
            cdf_plot(current, data, feature, type_of_monitors)
        elif feature == 'peeringDB_info_prefixes6':
            data['peeringDB_info_prefixes6'] = data.peeringDB_info_prefixes6.fillna(0)
            data['peeringDB_info_prefixes6'] = data.peeringDB_info_prefixes6.astype('Int64')
            cdf_plot(current, data, feature, type_of_monitors)
        elif feature == 'peeringDB_ix_count':
            data['peeringDB_ix_count'] = data.peeringDB_ix_count.fillna(0)
            data['peeringDB_ix_count'] = data.peeringDB_ix_count.astype('Int64')
            cdf_plot(current, data, feature, type_of_monitors)
        elif feature == 'peeringDB_fac_count':
            data['peeringDB_fac_count'] = data.peeringDB_fac_count.fillna(0)
            data['peeringDB_fac_count'] = data.peeringDB_fac_count.astype('Int64')
            cdf_plot(current, data, feature, type_of_monitors)
        elif feature == 'personal_is_matched':
            data['personal_is_matched'] = data.personal_is_matched.fillna(0)
            data['personal_is_matched'] = data.personal_is_matched.astype('Int64')
            histogram_plot(current, data, feature, type_of_monitors)
        elif feature == 'has_atlas_probe':
            data['has_atlas_probe'] = data.personal_is_matched.fillna(0)
            data['has_atlas_probe'] = data.personal_is_matched.astype('Int64')
            histogram_plot(current, data, feature, type_of_monitors)
        else:
            cdf_plot(current, data, feature, type_of_monitors)
            # cdf_subplot(ripe, data, feature)
    elif type == np.object:
        if feature == 'AS_rank_iso':
            # histogram_plot(ripe, data, feature)
            data['AS_rank_iso'] = convert_country_to_continent(data)
            histogram_plot(current, data, feature, type_of_monitors)
        elif feature == 'peeringDB_created':
            data['peeringDB_created'] = data.peeringDB_created.fillna(0)
            data['peeringDB_created'] = convert_to_numerical(data)
            cdf_plot(current, data, feature, type_of_monitors)
        elif feature == 'peeringDB_info_type':
            histogram_plot(current, data, feature, type_of_monitors)
        else:
            histogram_plot(current, data, feature, type_of_monitors)



def cdf_plot(unique_monitors, final, feature, monitors_origin):
    """
    :param unique_monitors: Contains the unique AS numbers of RIPE RIS or the AS numbers of RIPE ATLAS probes
    :param final: Contains a dataframe combining 4 datasets
    :param feature: Is the column name of final
    """
    x = final[feature].dropna()
    final_cdf = ECDF(x)
    plt.plot(final_cdf.x, final_cdf.y, label='All_ASes')
    merged_data = pd.merge(unique_monitors, final, on='ASn', how='inner')
    merged_data.sort_values('ASn', inplace=True)
    merged_data.drop_duplicates(subset='ASn', keep=False, inplace=True)
    merged_data.sort_values(feature, inplace=True)
    ripe_cdf = ECDF(merged_data[feature].dropna())
    plt.plot(ripe_cdf.x, ripe_cdf.y, label=monitors_origin)
    plt.ylabel('CDF')
    if feature == 'AS_rank_numberAddresses' or feature == 'AS_rank_numberAsns' or feature == 'AS_rank_numberPrefixes' \
            or feature == 'AS_rank_peer' or feature == 'AS_rank_provider' or feature == 'AS_rank_total' \
            or feature == 'ASn' or feature == 'AS_rank_customer' or feature == 'peeringDB_info_prefixes4' or \
            feature == 'peeringDB_info_prefixes6' or feature == 'peeringDB_ix_count' or feature == 'peeringDB_fac_count' \
            or feature == 'peeringDB_created':
        plt.xscale('log')
    else:
        plt.xscale('linear')
    plt.title('Feature: ' + str(feature), fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(feature) + str(monitors_origin) + f'.png')
    plt.show()


def histogram_plot(atlas, final, feature,  monitors_origin):
    """
    :param ripe: Contains the AS numbers of RIPE RIS
    :param final: Contains a dataframe combining 3 datasets
    :param feature: Is the column name of final
    """

    # Without dropna we pass all arguments except one (NaN) and the plots are all wrong
    x = final[feature].dropna()
    x = x.astype(str)
    merged_data = pd.merge(atlas, final, on=['ASn'], how='inner')
    y = merged_data[feature].astype(str)
    plt.hist((x, y), density=True, bins=final[feature].nunique(), histtype='bar', align='left',
             label=['All_ASes', monitors_origin],
             color=['blue', 'orange'])
    plt.legend(prop={'size': 10})
    plt.ylabel('CDF')
    plt.ylim(0, 1)
    plt.suptitle('Feature: ' + str(feature), fontsize=14)
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    plt.savefig(str(feature) + str(monitors_origin) + f'.png')
    plt.show()


def plot_analysis(dataset):
    """
    :param dataset: It contains all the datasets that will be plot
    """
    final_dataframe = read_final_dataframe()
    final_dataframe.rename(columns={'AS_rank_asn': 'ASn'}, inplace=True)
    for dt in dataset:
        if dt == 'Ripe_Ris_monitors':
            ripe_df = read_ripe_peers()
            type = 'RIPE_RIS_peers'
            call_categorize(final_dataframe, ripe_df, type)
        elif dt == 'Ripe_Atlas_probes':
            atlas_df = take_unique_ATLAS_ASNs()
            atlas_df.rename(columns={'Atlas_ASN': 'ASn'}, inplace=True)
            type = 'RIPE_ATLAS_probes'
            call_categorize(final_dataframe, atlas_df, type)
        else:
            raise Exception('Not defined type of dataset')