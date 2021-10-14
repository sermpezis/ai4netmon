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
    :param data: It contains all features from 3 different datasets
    :return:
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

    # convert list to dataframe
    df = pd.DataFrame(list_of_ripe_peers, columns=['ASn'])

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
    :param data:
    :param list_alpha_2:
    :return: Matches the acronyms with the Fullname of the countries
    """
    list_alpha_2 = [i.alpha2 for i in list(countries)]
    if data['AS_rank_iso'] in list_alpha_2:
        return pycountry.countries.get(alpha_2=data['AS_rank_iso']).name
    else:
        return 'Unknown Code'


def country_to_continent(country_name):
    try:
        country_alpha2 = pc.country_name_to_country_alpha2(country_name)
        country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
        return country_continent_name
    except:
        return 'No info'


def convert_country_to_continent(data):
    """
    :param data: Contains a dataframe combining 3 datasets
    :return: The continent for each country
    """
    # Convert iso = alpha_2 (example: US) to the whole name of the country (import iso3166)
    data['AS_rank_iso'] = data.apply(country_flag, axis=1)
    for i in range(0, len(data)):
        data['AS_rank_iso'][i] = country_to_continent(data['AS_rank_iso'][i])

    return data['AS_rank_iso']


def cdf_plot(ripe, final, feature):
    """
    :param ripe: Contains the AS numbers of RIPE RIS
    :param final: Contains a dataframe combining 3 datasets
    :param feature: Is the column name of final
    """
    final_cdf = ECDF(final[feature])
    plt.plot(final_cdf.x, final_cdf.y, label='All_ASes')
    merged_data = pd.merge(ripe, final, on='ASn', how='inner')
    merged_data.sort_values('ASn', inplace=True)
    merged_data.drop_duplicates(subset='ASn', keep=False, inplace=True)
    merged_data.sort_values(feature, inplace=True)
    ripe_cdf = ECDF(merged_data[feature])
    plt.plot(ripe_cdf.x, ripe_cdf.y, label='RIPE_RIS_peers')
    plt.ylabel('CDF')
    plt.xscale('log')
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
    merged_data = pd.merge(ripe, final, on='ASn', how='inner')
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
    x = final[feature].astype(str)
    merged_data = pd.merge(ripe, final, on=['ASn'], how='inner')
    y = merged_data[feature].astype(str)
    plt.hist([x, y], density=True, bins=final[feature].nunique(), histtype='bar', label=['All_ASes', 'RIPE_RIS_peers'],
             color=['blue', 'orange'])
    plt.legend(prop={'size': 10})
    plt.ylim(0, 1)
    plt.suptitle('Feature: ' + str(feature), fontsize=14)
    plt.xlabel('RIPE RIS peers')
    plt.xticks(rotation='vertical')
    plt.ylabel('CDF')
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
        if dataTypeObj == np.int64 or dataTypeObj == np.float64:
            if (column_name == 'personal_is_matched' or column_name == 'peeringDB_info_prefixes4' or column_name == 'peeringDB_info_prefixes6'\
                    or column_name == 'peeringDB_ix_count' or column_name == 'peeringDB_fac_count'):
                final_dataframe['personal_is_matched'] = final_dataframe.personal_is_matched.fillna(0)
                final_dataframe['personal_is_matched'] = final_dataframe.personal_is_matched.astype('Int64')
                cdf_plot(ripe_df, final_dataframe, column_name)
            else:
                cdf_plot(ripe_df, final_dataframe, column_name)
        elif dataTypeObj == np.object:
            ripe_sorted = ripe_df.sort_values(by=['ASn'], ascending=True)
            final_sorted = final_dataframe.sort_values(by=[column_name], ascending=True)
            if column_name == 'AS_rank_iso':
                histogram_plot(ripe_df, final_dataframe, column_name)
                final_dataframe['AS_rank_iso'] = convert_country_to_continent(final_dataframe)
                histogram_plot(ripe_sorted, final_dataframe, column_name)
            elif column_name == 'peeringDB_created':
                final_dataframe['peeringDB_created'] = final_dataframe.peeringDB_created.fillna(0)
                final_dataframe['peeringDB_created'] = convert_to_numerical(final_dataframe)
                cdf_plot(ripe_df, final_dataframe, column_name)
            else:
                histogram_plot(ripe_sorted, final_sorted, column_name)

