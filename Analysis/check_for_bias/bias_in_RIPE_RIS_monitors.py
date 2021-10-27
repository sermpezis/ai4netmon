import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from iso3166 import countries
import pycountry_convert as pc
import pycountry
import re
from datetime import datetime
from statsmodels.distributions.empirical_distribution import ECDF

pd.options.mode.chained_assignment = None
PATH_RIPE_RIS_PEERS = '../../Datasets/RIPE_RIS_peers_monitors/list_of_RIPE_RIS_peers.json'


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


def read_final_dataframe():
    """
    :return: A dataframe that is created from the concatenation of 3 datasets
    """

    return pd.read_csv('../aggregate_data/final_dataframe.csv', sep=',')


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
        return 'No info'


def convert_country_to_continent(data):
    """
    The function converts iso = alpha_2 (example: US) to the whole name of the country. Needs (import iso3166)
    :param data: Contains a dataframe combining 3 datasets
    :return: The continent for each country
    """
    data['AS_rank_iso'] = data.apply(country_flag, axis=1)
    for i in range(0, len(data)):
        data['AS_rank_iso'][i] = country_to_continent(data['AS_rank_iso'][i])

    return data['AS_rank_iso']


def categorize_features(data, ripe, type, feature):
    """
    :param data: Contains a dataframe combining 3 datasets
    :param ripe: A dataframe with one column that contains all RIPE peer-monitors ASn
    :param type: The dtypes of every column
    :param feature: Column name
    :return: Shows for every feature each plot
    """
    if type == np.int64 or type == np.float64:
        if feature == 'peeringDB_info_prefixes4':
            data['peeringDB_info_prefixes4'] = data.peeringDB_info_prefixes4.fillna(0)
            data['peeringDB_info_prefixes4'] = data.peeringDB_info_prefixes4.astype('Int64')
            cdf_plot(ripe, data, feature)
        elif feature == 'peeringDB_info_prefixes6':
            data['peeringDB_info_prefixes6'] = data.peeringDB_info_prefixes6.fillna(0)
            data['peeringDB_info_prefixes6'] = data.peeringDB_info_prefixes6.astype('Int64')
            cdf_plot(ripe, data, feature)
        elif feature == 'peeringDB_ix_count':
            data['peeringDB_ix_count'] = data.peeringDB_ix_count.fillna(0)
            data['peeringDB_ix_count'] = data.peeringDB_ix_count.astype('Int64')
            cdf_plot(ripe, data, feature)
        elif feature == 'peeringDB_fac_count':
            data['peeringDB_fac_count'] = data.peeringDB_fac_count.fillna(0)
            data['peeringDB_fac_count'] = data.peeringDB_fac_count.astype('Int64')
            cdf_plot(ripe, data, feature)
        elif feature == 'personal_is_matched':
            data['personal_is_matched'] = data.personal_is_matched.fillna(0)
            data['personal_is_matched'] = data.personal_is_matched.astype('Int64')
            histogram_plot(ripe, data, feature)
        else:
            cdf_plot(ripe, data, feature)
            # cdf_subplot(ripe, data, feature)
    elif type == np.object:
        if feature == 'AS_rank_iso':
            # histogram_plot(ripe, data, feature)
            data['AS_rank_iso'] = convert_country_to_continent(data)
            histogram_plot(ripe, data, feature)
            pass
        elif feature == 'peeringDB_created':
            final_dataframe['peeringDB_created'] = final_dataframe.peeringDB_created.fillna(0)
            final_dataframe['peeringDB_created'] = convert_to_numerical(data)
            cdf_plot(ripe, data, feature)
        elif feature == 'peeringDB_info_type':
            histogram_plot(ripe, data, feature)
        else:
            histogram_plot(ripe, data, feature)


def cdf_plot(ripe, final, feature):
    """
    :param ripe: Contains the AS numbers of RIPE RIS
    :param final: Contains a dataframe combining 3 datasets
    :param feature: Is the column name of final
    """
    x = final[feature].dropna()
    final_cdf = ECDF(x)
    plt.plot(final_cdf.x, final_cdf.y, label='All_ASes')
    merged_data = pd.merge(ripe, final, on='ASn', how='inner')
    merged_data.sort_values('ASn', inplace=True)
    merged_data.drop_duplicates(subset='ASn', keep=False, inplace=True)
    merged_data.sort_values(feature, inplace=True)
    ripe_cdf = ECDF(merged_data[feature].dropna())
    plt.plot(ripe_cdf.x, ripe_cdf.y, label='RIPE_RIS_peers')
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
    plt.savefig(str(feature) + f'.png')
    plt.show()


def cdf_subplot(ripe, final, feature):
    """
    :param ripe: Contains the AS numbers of RIPE RIS
    :param final: Contains a dataframe combining 3 datasets
    :param feature: Is the column name of final
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Feature: ' + str(feature), fontsize=14)
    ax1.set_xlabel('All ASns')
    ax1.set_ylabel('CDF')
    final_cdf = ECDF(final[feature])
    ax1.plot(final_cdf.x, final_cdf.y)
    merged_data = pd.merge(final, ripe, on='ASn', how='inner')
    merged_data.sort_values('ASn', inplace=True)
    merged_data.drop_duplicates(subset='ASn', keep=False, inplace=True)
    merged_data.sort_values(feature, inplace=True)
    ax2.set_xlabel('RIPE RIS peers')
    ripe_cdf = ECDF(merged_data[feature])
    ax2.plot(ripe_cdf.x, ripe_cdf.y)
    plt.tight_layout()
    plt.savefig(str(feature) + f'.png')
    plt.show()


def histogram_plot(ripe, final, feature):
    """
    :param ripe: Contains the AS numbers of RIPE RIS
    :param final: Contains a dataframe combining 3 datasets
    :param feature: Is the column name of final
    """
    # Without dropna we pass all arguments except one (NaN) and the plots are all wrong
    if feature == 'AS_rank_source':
        x = final[feature].fillna(value=feature)
    else:
        x = final[feature].dropna()
    x = x.astype(str)
    merged_data = pd.merge(ripe, final, on=['ASn'], how='inner')
    y = merged_data[feature].astype(str)
    plt.hist((x, y), density=True, bins=final[feature].nunique(), histtype='bar', align='left',
             label=['All_ASes', 'RIPE_RIS_peers'],
             color=['blue', 'orange'])
    plt.legend(prop={'size': 10})
    plt.ylabel('CDF')
    plt.ylim(0, 1)
    plt.suptitle('Feature: ' + str(feature), fontsize=14)
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    plt.savefig(str(feature) + f'.png')
    plt.show()


if __name__ == "__main__":

    ripe_df = read_ripe_peers()
    final_dataframe = read_final_dataframe()
    final_dataframe.rename(columns={'AS_rank_asn': 'ASn'}, inplace=True)
    print(final_dataframe.dtypes)
    for column_name in final_dataframe.columns:
        dataTypeObj = final_dataframe.dtypes[column_name]
        categorize_features(final_dataframe, ripe_df, dataTypeObj, column_name)