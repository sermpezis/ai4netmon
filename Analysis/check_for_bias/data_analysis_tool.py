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
PATH_RIPE_RIS_PEERS = '../../Datasets/RIPE_RIS_peers/list_of_RIPE_RIS_peers.json'
RIPE_ATLAS_PROBES = '../../Datasets/RIPE_Atlas_probes/bq_results.json'
ROUTEVIEWS_PEERS = '../../Datasets/RouteViews_peers/RouteViews-Peering-1_11_21.csv'


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


def take_unique_RouteViews_ASNs():
    """
    :return: The returned set contains only the unique ASns
    """

    data = pd.read_csv(ROUTEVIEWS_PEERS, sep=',')
    set1 = set(data['ASNUMBER'])

    route_views_dataframe = pd.DataFrame(set1)
    route_views_dataframe.columns = ['RouteViews_ASn']

    return route_views_dataframe


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


def call_categorize_all(final_df, ripe, atlas, route):

    for column_name in final_df.columns:
        dataTypeObj = final_df.dtypes[column_name]
        dcopy = final_df.copy(deep=True)
        dcopy1 = final_df.copy(deep=True)
        dcopy2 = final_df.copy(deep=True)
        if column_name == 'peeringDB_created':
            pass
        elif column_name == 'is_personal_AS':
            y1 = categorize_features_all(dcopy, ripe, dataTypeObj, column_name)
            y2 = categorize_features_all(dcopy, atlas, dataTypeObj, column_name)
            y3 = categorize_features_all(dcopy, route, dataTypeObj, column_name)
            x = final_df[column_name].fillna(0)
            x = x.astype(str)
            plt.hist([x, y1, y2, y3], density=True, bins=2, histtype='bar',
                     align='left',
                     label=['All_ASes', 'Ripe Ris', 'Atlas', 'RouteViews'])
            plt.legend(prop={'size': 10})
            plt.ylabel('CDF')
            plt.ylim(0, 1)
            plt.suptitle('Feature: ' + str(column_name), fontsize=14)
            plt.xticks(rotation='vertical')
            plt.tight_layout()
            plt.savefig(str(column_name) + 'All' + f'.png')
            plt.show()
        elif column_name == 'AS_rank_iso':
            y1 = categorize_features_all(dcopy, ripe, dataTypeObj, column_name)
            y2 = categorize_features_all(dcopy1, atlas, dataTypeObj, column_name)
            y3 = categorize_features_all(dcopy2, route, dataTypeObj, column_name)
            final_df[column_name] = convert_country_to_continent(final_df)
            x = final_df[column_name].dropna()
            x = x.astype(str)
            plt.hist([x, y1, y2, y3], density=True, bins=abs(final_df[column_name].nunique()), histtype='bar',
                     align='left',
                     label=['All_ASes', 'Ripe Ris', 'Atlas', 'RouteViews'])
            plt.legend(prop={'size': 10})
            plt.ylabel('CDF')
            plt.ylim(0, 1)
            plt.suptitle('Feature: ' + str(column_name), fontsize=14)
            plt.xticks(rotation='vertical')
            plt.tight_layout()
            plt.savefig(str(column_name) + 'All' + f'.png')
            plt.show()
        elif dataTypeObj == np.int64 or dataTypeObj == np.float64:
            x1, y1 = categorize_features_all(dcopy, ripe, dataTypeObj, column_name)
            x2, y2 = categorize_features_all(dcopy, atlas, dataTypeObj, column_name)
            x3, y3 = categorize_features_all(dcopy, route, dataTypeObj, column_name)
            x = final_df[column_name].dropna()
            final_cdf = ECDF(x)
            plt.plot(final_cdf.x, final_cdf.y, label='All_ASes')
            plt.plot(x1, y1, label='Ripe Ris')
            plt.plot(x2, y2, label='Atlas')
            plt.plot(x3, y3, label='RouteViews')
            plt.title('Feature: ' + str(column_name), fontsize=14)
            plt.legend()
            plt.tight_layout()
            plt.savefig(str(column_name) + 'All' + f'.png')
            plt.show()
        elif dataTypeObj == np.object:
            temp = final_df[column_name].dropna()
            x = temp.unique()
            x = x.astype(str)
            y0 = temp
            y0_counts = y0.value_counts()
            y1 = categorize_features_all(dcopy, ripe, dataTypeObj, column_name)
            c1 = 0
            c2 = 0
            c3 = 0
            for i in y1:
                if i == 'nan':
                    c1 = c1 + 1
            y1_counts = y1.value_counts()
            y2 = categorize_features_all(dcopy, atlas, dataTypeObj, column_name)
            for i in y2:
                if i == 'nan':
                    c2 = c2 + 1
            y2_counts = y2.value_counts()
            y3 = categorize_features_all(dcopy, route, dataTypeObj, column_name)
            for i in y3:
                if i == 'nan':
                    c3 = c3 + 1
            y3_counts = y3.value_counts()

            y0_list = []
            y1_list = []
            y2_list = []
            y3_list = []
            for item in x:
                y0_value = 0
                y1_value = 0
                y2_value = 0
                y3_value = 0
                if item in y0_counts:
                    y0_value = y0_counts[item]
                if item in y1_counts:
                    y1_value = y1_counts[item]
                if item in y2_counts:
                    y2_value = y2_counts[item]
                if item in y3_counts:
                    y3_value = y3_counts[item]
                y0_list.append(y0_value/len(y0))
                y1_list.append(y1_value/(len(y1) - c1))
                y2_list.append(y2_value/(len(y2) - c2))
                y3_list.append(y3_value/(len(y3) - c3))
            bar_width = 0.2
            x_1 = np.arange(len(x))
            x_2 = [x + bar_width for x in x_1]
            x_3 = [x + bar_width for x in x_2]
            x_4 = [x + bar_width for x in x_3]
            plt.bar(x_1, y0_list, label='All ASes', width=bar_width)
            plt.bar(x_2, y1_list, label='Ripe Ris', width=bar_width)
            plt.bar(x_3, y2_list, label='ATLAS', width=bar_width)
            plt.bar(x_4, y3_list, label='RouteView', width=bar_width)

            plt.legend(prop={'size': 10})
            plt.ylabel('CDF')
            plt.ylim(0, 1)
            plt.suptitle('Feature: ' + str(column_name), fontsize=14)
            plt.xticks([r + bar_width for r in range(len(x))], x, rotation='vertical')
            plt.tight_layout()
            plt.savefig(str(column_name) + 'All' + f'.png')
            plt.show()
            plt.close()



def convert_to_numerical(data):
    """
    The function subtracts the created year of peeringDB from the current year.
    :param data: It contains all features from 3 different datasets
    :return: A numerical feature containing the above described subtraction
    """
    data['peeringDB_created'] = data['peeringDB_created'].astype('str')
    data['peeringDB_created'] = data['peeringDB_created'].apply(lambda x: keep_number(x))
    today_year = datetime.today()
    data['peeringDB_created'] = data['peeringDB_created'].apply(lambda x: (int(today_year.year)) - int(x))

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
    data['AS_rank_iso'] = df['AS_rank_iso']

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
        elif feature == 'is_personal_AS':
            data['is_personal_AS'] = data['is_personal_AS'].replace('', np.nan)
            data['is_personal_AS'] = data.is_personal_AS.astype('Int64')
            histogram_plot(current, data, feature, type_of_monitors)
        elif feature == 'has_atlas_probe':
            data['has_atlas_probe'] = data.has_atlas_probe.fillna(0)
            data['has_atlas_probe'] = data.has_atlas_probe.astype('Int64')
            histogram_plot(current, data, feature, type_of_monitors)
        else:
            cdf_plot(current, data, feature, type_of_monitors)
            # cdf_subplot(ripe, data, feature)
    elif type == np.object:
        if feature == 'AS_rank_iso':
            data['AS_rank_iso'] = convert_country_to_continent(data)
            histogram_plot(current, data, feature, type_of_monitors)
        elif feature == 'peeringDB_created':
            data['peeringDB_created'] = data.peeringDB_created.fillna(0)
            data['peeringDB_created'] = convert_to_numerical(data)
            cdf_plot(current, data, feature, type_of_monitors)
        elif feature == 'peeringDB_info_type':
            histogram_plot(current, data, feature, type_of_monitors)
        elif feature == 'AS_rank_source':
            data['AS_rank_source'].fillna(np.nan, inplace=True)
            histogram_plot(current, data, feature, type_of_monitors)
        else:
            histogram_plot(current, data, feature, type_of_monitors)


def categorize_features_all(data, current, type, feature):

    if type == np.int64 or type == np.float64:
        if feature == 'peeringDB_info_prefixes4':
            data['peeringDB_info_prefixes4'] = data.peeringDB_info_prefixes4.fillna(0)
            data['peeringDB_info_prefixes4'] = data.peeringDB_info_prefixes4.astype('Int64')
            x, y = cdf_plot_all(current, data, feature)
            return x, y
        elif feature == 'peeringDB_info_prefixes6':
            data['peeringDB_info_prefixes6'] = data.peeringDB_info_prefixes6.fillna(0)
            data['peeringDB_info_prefixes6'] = data.peeringDB_info_prefixes6.astype('Int64')
            x, y = cdf_plot_all(current, data, feature)
            return x, y
        elif feature == 'peeringDB_ix_count':
            data['peeringDB_ix_count'] = data.peeringDB_ix_count.fillna(0)
            data['peeringDB_ix_count'] = data.peeringDB_ix_count.astype('Int64')
            x, y = cdf_plot_all(current, data, feature)
            return x, y
        elif feature == 'peeringDB_fac_count':
            data['peeringDB_fac_count'] = data.peeringDB_fac_count.fillna(0)
            data['peeringDB_fac_count'] = data.peeringDB_fac_count.astype('Int64')
            x, y = cdf_plot_all(current, data, feature)
            return x, y
        elif feature == 'AS_hegemony':
            data['AS_hegemony'] = data.AS_hegemony.replace('', np.nan)
            data['AS_hegemony'] = data.AS_hegemony.astype(float)
            x, y = cdf_plot_all(current, data, feature)
            return x, y
        elif feature == 'is_personal_AS':
            data['is_personal_AS'] = data['is_personal_AS'].replace('', np.nan)
            data['is_personal_AS'] = data.is_personal_AS.astype('string')
            y = histogram_plot_all(current, data, feature)
            return y
        elif feature == 'has_atlas_probe':
            data['has_atlas_probe'] = data.has_atlas_probe.fillna(0)
            data['has_atlas_probe'] = data.has_atlas_probe.astype('Int64')
            y = histogram_plot_all(current, data, feature)
            return y
        else:
            x, y = cdf_plot_all(current, data, feature)
            return x, y
    elif type == np.object:
        if feature == 'AS_rank_iso':
            data['AS_rank_iso'] = convert_country_to_continent(data)
            y = histogram_plot_all(current, data, feature)
            return y
        elif feature == 'peeringDB_created':
            data['peeringDB_created'] = data.peeringDB_created.fillna(0)
            data['peeringDB_created'] = convert_to_numerical(data)
            x, y = cdf_plot_all(current, data, feature)
            return x, y
        elif feature == 'peeringDB_info_type':
            y = histogram_plot_all(current, data, feature)
            return y
        elif feature == 'AS_rank_source':
            data['AS_rank_source'].fillna(np.nan, inplace=True)
            y = histogram_plot_all(current, data, feature)
            return y
        else:
            y = histogram_plot_all(current, data, feature)
            return y


def cdf_plot_all(unique_monitors, final, feature):
    """
    :param unique_monitors: Contains the unique AS numbers of RIPE RIS or the AS numbers of RIPE ATLAS probes
    :param final: Contains a dataframe combining 4 datasets
    :param feature: Is the column name of final
    """
    merged_data = pd.merge(unique_monitors, final, on='ASn', how='inner')
    merged_data.sort_values('ASn', inplace=True)
    merged_data.drop_duplicates(subset='ASn', keep=False, inplace=True)
    merged_data.sort_values(feature, inplace=True)
    ripe_cdf = ECDF(merged_data[feature].dropna())
    plt.ylabel('CDF')
    if feature == 'AS_rank_numberAddresses' or feature == 'AS_rank_numberAsns' or feature == 'AS_rank_numberPrefixes' \
            or feature == 'AS_rank_peer' or feature == 'AS_rank_provider' or feature == 'AS_rank_total' \
            or feature == 'ASn' or feature == 'AS_rank_customer' or feature == 'peeringDB_info_prefixes4' or \
            feature == 'peeringDB_info_prefixes6' or feature == 'peeringDB_ix_count' or feature == 'peeringDB_fac_count' \
            or feature == 'peeringDB_created' or feature == 'AS_hegemony':
        plt.xscale('log')
    else:
        plt.xscale('linear')

    return ripe_cdf.x, ripe_cdf.y


def histogram_plot_all(unique_monitors, final, feature):
    """
    :param monitors_origin:
    :param ripe: Contains the AS numbers of RIPE RIS
    :param final: Contains a dataframe combining 3 datasets
    :param feature: Is the column name of final
    """
    # Without dropna we pass all arguments except one (NaN) and the plots are all wrong

    merged_data = pd.merge(unique_monitors, final, on=['ASn'], how='inner')
    y = merged_data[feature].astype(str)

    return y


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


def histogram_plot(unique_monitors, final, feature, monitors_origin):
    """
    :param monitors_origin:
    :param ripe: Contains the AS numbers of RIPE RIS
    :param final: Contains a dataframe combining 3 datasets
    :param feature: Is the column name of final
    """
    # Without dropna we pass all arguments except one (NaN) and the plots are all wrong
    x = final[feature].dropna()
    x = x.astype(str)
    merged_data = pd.merge(unique_monitors, final, on=['ASn'], how='inner')
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
    final_dataframe.rename(columns={'ASN': 'ASn'}, inplace=True)
    for dt in dataset:
        dd = final_dataframe.copy(deep=True)
        if dt == 'Ripe_Ris_monitors':
            ripe_df = read_ripe_peers()
            type = 'RIPE_RIS_peers'
            call_categorize(dd, ripe_df, type)
        elif dt == 'Ripe_Atlas_probes':
            atlas_df = take_unique_ATLAS_ASNs()
            atlas_df.rename(columns={'Atlas_ASN': 'ASn'}, inplace=True)
            type = 'RIPE_ATLAS_probes'
            call_categorize(dd, atlas_df, type)
        elif dt == 'RouteViews_peers':
            route_df = take_unique_RouteViews_ASNs()
            route_df.rename(columns={'RouteViews_ASn': 'ASn'}, inplace=True)
            type = 'RouteViews_peers'
            call_categorize(dd, route_df, type)
        elif dt == 'Compare_All':
            ripe = read_ripe_peers()
            atlas = take_unique_ATLAS_ASNs()
            atlas.rename(columns={'Atlas_ASN': 'ASn'}, inplace=True)
            route = take_unique_RouteViews_ASNs()
            route.rename(columns={'RouteViews_ASn': 'ASn'}, inplace=True)
            call_categorize_all(dd, ripe, atlas, route)
        else:
            raise Exception('Not defined type of dataset')
