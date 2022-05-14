from ai4netmon.Analysis.bias import bias_utils as bu
from ai4netmon.Analysis.bias import radar_chart
from ai4netmon.Analysis.aggregate_data import data_aggregation_tools as dat
# from ai4netmon.Analysis.bias import generate_distribution_plots as gdp
import pandas as pd

## set variables
FIG_RADAR_FNAME_FORMAT = './figures/fig_radar_{}.png'
SAVE_PLOTS_DISTRIBUTION_FNAME_FORMAT = './figures/Fig_{}_{}'

## select features for visualization
FEATURE_NAMES_DICT = bu.get_features_dict_for_visualizations() 
FEATURES = list(FEATURE_NAMES_DICT.keys())

## load data
df = dat.load_aggregated_dataframe(preprocess=True)

## select infrastructure
df_ris = df.loc[(df['is_ris_peer_v4']>0) | (df['is_ris_peer_v6']>0)]
df_ris_v4 = df.loc[(df['is_ris_peer_v4']>0)]
df_ris_v6 = df.loc[(df['is_ris_peer_v6']>0)]
ris_asns = list(df_ris.index)
df_rv = df.loc[df['is_routeviews_peer']>0]
rv_asns = list(df_rv.index)
df_atlas = df.loc[(df['nb_atlas_probes_v4']>0) | (df['nb_atlas_probes_v6']>0)]
df_atlas_v4 = df.loc[(df['nb_atlas_probes_v4']>0)]
df_atlas_v6 = df.loc[(df['nb_atlas_probes_v6']>0)]
atlas_asns = list(df_atlas.index)
df_ris_rv = df.loc[(df['is_ris_peer_v4']>0) | (df['is_ris_peer_v6']>0) | (df['is_routeviews_peer']>0)]
ris_rv_asns = list(df_ris_rv.index)

def get_full_feeders():
    STATS_FILE = 'https://github.com/LarsxGitHub/feed_stats/blob/main/features_per_asn.pkl?raw=true'
    df = pd.read_pickle(STATS_FILE)
    return df[df['origins4'] > 70000].index.tolist()
full_feeders = get_full_feeders()

## create dicts for input to bias method
network_sets_dict = dict()
network_sets_dict['all'] = df
network_sets_dict['Atlas'] = df_atlas
network_sets_dict['Atlas (IPv4)'] = df_atlas_v4
network_sets_dict['Atlas (IPv6)'] = df_atlas_v6
network_sets_dict['RIS'] = df_ris
network_sets_dict['RIS (IPv4)'] = df_ris_v4
network_sets_dict['RIS (IPv6)'] = df_ris_v6
network_sets_dict['RIS (full feeders)'] = df.loc[set(ris_asns).intersection(full_feeders),:]
network_sets_dict['RV'] = df_rv
network_sets_dict['RV (full feeders)'] = df.loc[set(rv_asns).intersection(full_feeders),:]
network_sets_dict['RIS & RV'] = df_ris_rv
network_sets_dict['RIS & RV (full feeders)'] = df.loc[set(ris_rv_asns).intersection(full_feeders),:]


for k,v in network_sets_dict.items():
    print('{}\t {} peers'.format(k,v.shape[0]))

network_sets_dict_for_bias = {k:v[FEATURES] for k,v in network_sets_dict.items() if k != 'all'}

## calculate bias
params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}
bias_df = bu.bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **params)

## print biases
print('Bias per monitor set (columns) and per feature (rows)')
print_df = bias_df.copy()
print_df.index = [n.replace('\n','') for n in FEATURE_NAMES_DICT.values()]
print(print_df.round(2))


## plot radar plots
FONTSIZE = 15
FONTSIZE_SMALL = 12
plot_sets = ['RIS', 'RV', 'RIS & RV']
radar_chart.plot_radar_from_dataframe(bias_df[plot_sets], colors=None, frame='polygon',
    legend_loc=(0.75, .95), fontsize=FONTSIZE, fontsize_features=FONTSIZE_SMALL,
    save_filename=FIG_RADAR_FNAME_FORMAT.format('only_RCs'), varlabels=FEATURE_NAMES_DICT)
plot_sets = ['RIS', 'RV', 'RIS (full feeders)', 'RV (full feeders)']
radar_chart.plot_radar_from_dataframe(bias_df[plot_sets], colors=None, frame='polygon',
    legend_loc=(0.75, .95), fontsize=FONTSIZE, fontsize_features=FONTSIZE_SMALL,
    save_filename=FIG_RADAR_FNAME_FORMAT.format('only_RCs_full_feeders'), varlabels=FEATURE_NAMES_DICT)
plot_sets = ['RIS', 'RIS (IPv4)', 'RIS (IPv6)']
radar_chart.plot_radar_from_dataframe(bias_df[plot_sets], colors=None, frame='polygon',
    legend_loc=(0.75, .95), fontsize=FONTSIZE, fontsize_features=FONTSIZE_SMALL,
    save_filename=FIG_RADAR_FNAME_FORMAT.format('RIS_v4_v6'), varlabels=FEATURE_NAMES_DICT)
plot_sets = ['Atlas', 'Atlas (IPv4)', 'Atlas (IPv6)']
radar_chart.plot_radar_from_dataframe(bias_df[plot_sets], colors=None, frame='polygon',
    legend_loc=(0.75, .95), fontsize=FONTSIZE, fontsize_features=FONTSIZE_SMALL,
    save_filename=FIG_RADAR_FNAME_FORMAT.format('Atlas_v4_v6'), varlabels=FEATURE_NAMES_DICT)