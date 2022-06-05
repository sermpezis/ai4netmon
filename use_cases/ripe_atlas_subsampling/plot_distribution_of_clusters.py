import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

KMEANS_10 = './data/clusters_lists_of_Kmeans_10_0.csv'
KMEANS_20 = './data/clusters_lists_of_Kmeans_20_0.csv'
GREEDY_LEAST = './data/selected_from_greedy_least_similar.csv'

OPTIMAL_RIPE_RIS = 'https://raw.githubusercontent.com/sermpezis/ai4netmon/main/use_cases/ripe_ris_subsampling/data/sorted_list_greedy_Atlas.json'

SPECTRAL_10 = './data/clusters_lists_of_SpectralClustering_10_0.csv'
SPECTRAL_20 = './data/clusters_lists_of_SpectralClustering_20_0.csv'
SPECTRAL_100 = './data/clusters_lists_of_SpectralClustering_100_0.csv'

FIG_SAVE_FNAME = './distribution_in_clusters/distribution_in_{}_{}_for_{}_monitors_based_on_{}.png'

FONTSIZE = 15

kmeans_10_clusters = pd.read_csv(KMEANS_10)
kmeans_10_clusters = kmeans_10_clusters.drop(['Unnamed: 0'], axis=1)
kmeans_20_clusters = pd.read_csv(KMEANS_20)
kmeans_20_clusters = kmeans_20_clusters.drop(['Unnamed: 0'], axis=1)

spectral_10_clusters = pd.read_csv(SPECTRAL_10)
spectral_10_clusters = spectral_10_clusters.drop(['Unnamed: 0'], axis=1)
spectral_20_clusters = pd.read_csv(SPECTRAL_20)
spectral_20_clusters = spectral_20_clusters.drop(['Unnamed: 0'], axis=1)
spectral_100_clusters = pd.read_csv(SPECTRAL_100)
spectral_100_clusters = spectral_100_clusters.drop(['Unnamed: 0'], axis=1)

greedy_least = pd.read_csv(GREEDY_LEAST)
greedy_least_asns = greedy_least.iloc[:, 1].to_list()

optimal_ripe_ris = pd.read_json(OPTIMAL_RIPE_RIS)
optimal_ripe_ris = optimal_ripe_ris.iloc[:,0].to_list()
optimal_ripe_ris_asns = [int(float(i)) for i in optimal_ripe_ris]

def calculate_dist_and_plot(clusters, optimal_asns, clustering_method, optimal_asns_name):
    """
    Function that iterates through asns of optimal list and counts how many of those asns
    are inside every cluster of the kmeans algorithm.
    param: clusters: the df with the clusters and the asns they contain
    """
    number_of_clusters = clusters.shape[1]
    count_dist = []
    range_of_asns = [50, 100, 300, 1000]
    # because optimal asns are reversed
    if optimal_asns_name == 'optimal_ripe_ris_asns':
        range_of_asns.reverse()
    for i in range(len(range_of_asns)):
        count_ = []
        for asn_of_cluster in range(number_of_clusters):
            counter = 0

            for asn in range(range_of_asns[i]):

                if optimal_asns[asn] in clusters[str(asn_of_cluster)].values:
                    counter += clusters[str(asn_of_cluster)].value_counts()[optimal_asns[asn]]
            count_.append(counter)
        count_dist.append(count_)

    print(count_dist)

    x_vals = list(range(number_of_clusters))

    for i in range(len(range_of_asns)):

        plt.plot(x_vals, count_dist[i])
        plt.xticks(range(0, number_of_clusters, 2), fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        plt.xlabel('Clusters', fontsize=FONTSIZE)
        plt.ylabel('#monitors', fontsize=FONTSIZE)
        plt.grid(True)
        plt.subplots_adjust(left=0.15, bottom=0.15)
        plt.savefig(FIG_SAVE_FNAME.format(clustering_method, number_of_clusters, range_of_asns[i], optimal_asns_name))
        plt.close()


