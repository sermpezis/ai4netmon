import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

PATH_RIPE_RIS_PEERS = '../../Datasets/RIPE_RIS_peers_monitors/list_of_RIPE_RIS_peers.json'

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

def cdf_plot(ripe, final, feature):
    """
    :param ripe: Contains the AS numbers of RIPE RIS
    :param final: Contains a dataframe combining 3 datasets
    :param feature: Is the column name of final
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Feature: ' + str(feature), fontsize=14)
    ax1.set_xlabel('Final data')
    ax1.set_ylabel('CDF Analysis')
    final_cdf = ECDF(final[feature])
    ax1.plot(final_cdf.x, final_cdf.y)
    merged_data = pd.merge(ripe, final, on='ASn', how='inner')
    merged_data.sort_values('ASn', inplace=True)
    merged_data.drop_duplicates(subset='ASn', keep=False, inplace=True)
    merged_data.sort_values(feature, inplace=True)
    ax2.set_xlabel('Ripe data')
    ripe_cdf = ECDF(merged_data[feature])
    ax2.plot(ripe_cdf.x, ripe_cdf.y)
    plt.savefig(str(feature) + f'.png')
    plt.tight_layout()
    plt.show()

def histogram_plot(ripe, final, feature):
    """
    :param ripe: Contains the AS numbers of RIPE RIS
    :param final: Contains a dataframe combining 3 datasets
    :param feature: Is the column name of final
    """
    plt.style.use('seaborn-deep')
    x = final[feature].astype(str)
    merged_data = pd.merge(ripe, final, on=['ASn'], how='inner')
    y = merged_data[feature].astype(str)
    plt.hist([x, y], density=True, bins=10, histtype='bar', label=['final', 'ripe'], color=['blue', 'orange'])
    plt.legend(prop={'size': 10})
    plt.suptitle('Feature: ' + str(feature), fontsize=14)
    plt.xlabel('Ripe data')
    plt.xticks(rotation='vertical')
    plt.ylabel('CDF Analysis')
    plt.savefig(str(feature) + f'.png')
    plt.tight_layout()
    plt.show()

def test_CDF_plots(ripe, final, feature):
    """
    :param ripe: Contains the AS numbers of RIPE RIS
    :param final: Contains a dataframe combining 3 datasets
    :param feature: Is the column name of final
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Feature: ' + str(feature), fontsize=14)
    print(final[feature])
    my_cdf = ECDF(final[feature])
    ax1.plot(my_cdf.x, my_cdf.y)

    merged_data = pd.merge(ripe, final, on='ASn', how='inner')
    merged_data.sort_values('ASn', inplace=True)
    merged_data.drop_duplicates(subset='ASn', keep=False, inplace=True)
    merged_data.sort_values(feature, inplace=True)
    ripe_cdf = ECDF(merged_data[feature])
    ax2.plot(ripe_cdf.x, ripe_cdf.y)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    ripe_df = read_ripe_peers()
    final_dataframe = read_final_dataframe()
    final_dataframe.rename(columns={'AS_rank_asn': 'ASn'}, inplace=True)
    for column_name in final_dataframe.columns:

        dataTypeObj = final_dataframe.dtypes[column_name]
        if dataTypeObj == np.int64 or dataTypeObj == np.float64:
            ripe_sorted = ripe_df.sort_values(by=['ASn'], ascending=True)
            final_sorted = final_dataframe.sort_values(by=[column_name], ascending=True)
            cdf_plot(ripe_sorted, final_sorted, column_name)
            # test_CDF_plots(ripe_sorted, final_sorted, column_name)
        elif dataTypeObj == np.object:
            ripe_sorted = ripe_df.sort_values(by=['ASn'], ascending=True)
            final_sorted = final_dataframe.sort_values(by=[column_name], ascending=True)
            histogram_plot(ripe_sorted, final_sorted, column_name)

