from ai4netmon.Analysis.bias import bias_utils as bu
from ai4netmon.Analysis.bias import radar_chart
from ai4netmon.Analysis.aggregate_data import data_aggregation_tools as dat
from ai4netmon.Analysis.bias import generate_distribution_plots as gdp
import pandas as pd
import re

# choose simmilarity matrix to sample from, Ripe Atlas asn similarity or probe similarity
# 'ASN' for the first option, 'PROBES' otherwise

SIMMATRIX = 'ASN'
if SIMMATRIX == 'PROBES':

    ## datasets
    FIG_RADAR_FNAME_FORMAT = './figures/Fig_SOFIA_radar_asns_from_probes{}.png'
    SAVE_PLOTS_DISTRIBUTION_FNAME_FORMAT = './figures/Fig_SOFIA_asns_from_probes_{}_{}'

    # datasets
    KMEANS_10 = './data/selected_from_kmeans10_asns_of_probes.csv'
    KMEANS_20 = './data/selected_from_kmeans20_asns_of_probes.csv'
    GREEDY_LEAST = './data/selected_from_greedy_least_asns_of_probes.csv'

elif SIMMATRIX == 'ASN':

    ## datasets
    FIG_RADAR_FNAME_FORMAT = './figures/Fig_SOFIA_radar_{}.png'
    SAVE_PLOTS_DISTRIBUTION_FNAME_FORMAT = './figures/Fig_SOFIA_{}_{}'

    # datasets
    KMEANS_10 = './data/selected_from_k_means_10_mean.csv'
    KMEANS_20 = './data/selected_from_k_means_20_mean.csv'
    GREEDY_LEAST = './data/selected_from_greedy_least_similar_mean.csv'

# select features for visualization
FEATURE_NAMES_DICT = bu.get_features_dict_for_visualizations()
FEATURES = list(FEATURE_NAMES_DICT.keys())

## load data
df = dat.load_aggregated_dataframe(preprocess=True)

## load first list of selected asns with clustering and greedy methods.

kmeans_10 = pd.read_csv(KMEANS_10)
kmeans_20 = pd.read_csv(KMEANS_20)
greedy_least = pd.read_csv(GREEDY_LEAST)

# keep one column of sampled asns and drop nan values

kmeans_10list = kmeans_10.iloc[:, 1]
kmeans_10list = kmeans_10list.dropna()

kmeans_20list = kmeans_20.iloc[:, 1]
kmeans_20list = kmeans_20list.dropna()

greedy_least_list = greedy_least.iloc[:, 1]
greedy_least_list = greedy_least_list.dropna()

# turn pandas series into lists
kmeans_10list = list(kmeans_10list)
kmeans_20list = list(kmeans_20list)
greedy_least_list = list(greedy_least_list)

# put lists inside a list and iterate it to check whether there are indices in the lists of sampled asns
# that do not exist in the whole df

sampled_lists = [kmeans_10list, kmeans_20list, greedy_least_list]

for list_ in sampled_lists:
    try:
        df.loc[list_]
    except KeyError as e:
        idx = re.search(r"\[([A-Za-z0-9_.]+)\]", str(e))
        if idx is not None:
            if '.' in idx.group(1):
                list_.remove(float(idx.group(1)))
            else:
                list_.remove(int(idx.group(1)))

# create dict with sets
network_sets_dict = dict()
network_sets_dict['all'] = df
network_sets_dict['RIPE Atlas'] = df.loc[(df['nb_atlas_probes_v4']>0) | (df['nb_atlas_probes_v6']>0)]
network_sets_dict['KMEANS 10'] = df.loc[kmeans_10list]
network_sets_dict['KMEANS 20'] = df.loc[kmeans_20list]
network_sets_dict['GREEDY LEAST'] = df.loc[greedy_least_list]


network_sets_dict_for_bias = {k:v[FEATURES] for k,v in network_sets_dict.items() if k != 'all'}

params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}
bias_df = bu.bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **params)

# print biases & save to csv
print('Bias per monitor set (columns) and per feature (rows)')
print_df = bias_df.copy()
print_df.index = [n.replace('\n','') for n in FEATURE_NAMES_DICT.values()]
print(print_df.round(2))
    
# plot radar plot
plot_df = bias_df[['RIPE Atlas', 'KMEANS 10', 'KMEANS 20', 'GREEDY LEAST']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', save_filename=FIG_RADAR_FNAME_FORMAT.format('all'), varlabels=FEATURE_NAMES_DICT)


## plot detailed distibutions
network_sets_dict_plots = {'All ASes': network_sets_dict['all'], 
                     'RIPE Atlas': network_sets_dict['RIPE Atlas'], 
                     'KMEANS 10': network_sets_dict['KMEANS 10'],
                     'KMEANS 20': network_sets_dict['KMEANS 20'],
                           'GREEDY LEAST': network_sets_dict['GREEDY LEAST']}
gdp.plot_all(network_sets_dict_plots, SAVE_PLOTS_DISTRIBUTION_FNAME_FORMAT, save_json=False, show_plot=True)
