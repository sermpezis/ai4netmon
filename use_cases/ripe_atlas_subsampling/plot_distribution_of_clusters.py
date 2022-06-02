import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

KMEANS_10 = './data/clusters_lists_of_Kmeans_10_0.csv'
KMEANS_20 = './data/clusters_lists_of_Kmeans_20_0.csv'
GREEDY_LEAST = './data/selected_from_greedy_least_similar.csv'

FIG_SAVE_FNAME = './distribution_in_clusters/distribution_in_kmeans_{}_for_{}_monitors.png'
FONTSIZE = 15

kmeans_10_clusters = pd.read_csv(KMEANS_10)
kmeans_10_clusters = kmeans_10_clusters.drop(['Unnamed: 0'], axis=1)
kmeans_20_clusters = pd.read_csv(KMEANS_20)
kmeans_20_clusters = kmeans_20_clusters.drop(['Unnamed: 0'], axis=1)


greedy_least = pd.read_csv(GREEDY_LEAST)

greedy_least_asns = greedy_least.iloc[:, 1].to_list()


def calculate_dist_and_plot(clusters):
    """
    Function that iterates through asns of greedy method (optimal) and counts how many of those asns
    are inside every cluster of the kmeans algorithm.
    param: clusters: the df with the clusters and the asns they contain
    """
    number_of_clusters = clusters.shape[1]
    count_dist = []
    range_of_asns = [50, 100, 300, 1000]
    for i in range(len(range_of_asns)):
        count_ = []
        for asn_of_cluster in range(number_of_clusters):
            counter = 0

            for asn in range(range_of_asns[i]):

                if greedy_least_asns[asn] in clusters[str(asn_of_cluster)].values:
                    counter += clusters[str(asn_of_cluster)].value_counts()[greedy_least_asns[asn]]
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
        plt.savefig(FIG_SAVE_FNAME.format(number_of_clusters, range_of_asns[i]))
        plt.close()


