import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PATH_RIPE_RIS_PEERS = '../Datasets/RIPE_RIS_peers_monitors/list_of_RIPE_RIS_peers.json'

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

    return pd.read_csv('aggregate_data/final_dataframe.csv', sep=',')


def cdf_plot(ripe, final, feature):

    x_final_sorted = np.linspace(0, 1, len(final))
    x_ripe_sorted = np.linspace(0, 1, len(ripe))

    fig = plt.figure()
    plt.title('Feature: ' + str(feature), fontsize=14, pad=20)
    ax1 = fig.add_subplot(121)
    ax1.plot(x_final_sorted, final[feature])
    ax1.set_xlabel('Final data')
    ax1.set_ylabel('CDF Analysis')

    ax2 = fig.add_subplot(122)
    merged_data = pd.merge(ripe, final, on=['ASn'], how='inner')
    ax2.plot(x_ripe_sorted, merged_data[feature])
    ax2.set_xlabel('Ripe data')
    ax2.set_ylabel('CDF Analysis')
    # plt.savefig(str(feature) + f'.png')
    plt.tight_layout()
    plt.show()
    plt.clf()


def histogram_plot(ripe, final, feature):

    plt.style.use('seaborn-deep')
    x = final[feature].astype(str)
    merged_data = pd.merge(ripe, final, on=['ASn'], how='inner')
    y = merged_data[feature].astype(str)
    plt.hist(x, cumulative=True, density=1, bins=1000, histtype='step', label='final')
    plt.hist(y, cumulative=True, density=1, bins=1000, histtype='step', label='ripe')
    plt.legend(loc='upper right')
    plt.suptitle('Feature: ' + str(feature), fontsize=14)
    plt.xlabel('Ripe data')
    plt.xticks(rotation='vertical')
    plt.ylabel('CDF Analysis')
    plt.savefig(str(feature) + f'.png')
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
    elif dataTypeObj == np.object:
        ripe_sorted = ripe_df.sort_values(by=['ASn'], ascending=True)
        final_sorted = final_dataframe.sort_values(by=[column_name], ascending=True)
        # histogram_plot(ripe_sorted, final_sorted, column_name)


