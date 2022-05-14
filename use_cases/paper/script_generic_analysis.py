from ai4netmon.Analysis.bias import bias_utils as bu
from ai4netmon.Analysis.bias import radar_chart
from ai4netmon.Analysis.aggregate_data import data_aggregation_tools as dat
from ai4netmon.Analysis.bias import generate_distribution_plots as gdp
# import matplotlib.colors as mcolors


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
ris_asns = list(df_ris.index)
df_rv = df.loc[df['is_routeviews_peer']>0]
rv_asns = list(df_rv.index)
df_atlas = df.loc[(df['nb_atlas_probes_v4']>0) | (df['nb_atlas_probes_v6']>0)]
atlas_asns = list(df_atlas.index)
df_ris_rv = df.loc[(df['is_ris_peer_v4']>0) | (df['is_ris_peer_v6']>0) | (df['is_routeviews_peer']>0)]
ris_rv_asns = list(df_ris_rv.index)

## create dicts for input to bias method
network_sets_dict = dict()
network_sets_dict['all'] = df
network_sets_dict['RIPE Atlas'] = df_atlas
network_sets_dict['RIPE RIS'] = df_ris
network_sets_dict['RouteViews'] = df_rv
network_sets_dict['RIPE RIS & RouteViews'] = df_ris_rv

network_sets_dict_for_bias = {k:v[FEATURES] for k,v in network_sets_dict.items() if k != 'all'}

## calculate bias
params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}
bias_df = bu.bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **params)
params={'method':'max_variation', 'bins':10}
bias_df_max = bu.bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **params)
params={'method':'total_variation', 'bins':10}
bias_df_tv = bu.bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **params)
params={'method':'max_variation', 'bins':10}
bias_df_max = bu.bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **params)
params={'method':'ks_test'}
bias_df_ks = bu.bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **params)

## print biases
print('Bias per monitor set (columns) and per feature (rows)')
print_df = bias_df.copy()
print_df.index = [n.replace('\n','') for n in FEATURE_NAMES_DICT.values()]
print(print_df.round(2))

print('KS-test p-value per monitor set (columns) and per feature (rows)')
print_df = bias_df_ks.copy()
print_df.index = [n.replace('\n','') for n in FEATURE_NAMES_DICT.values()]
print(print_df.round(2))


## plot radar plots
FONTSIZE = 15
FONTSIZE_SMALL = 12
plot_sets = ['RIPE Atlas','RIPE RIS', 'RouteViews']
radar_chart.plot_radar_from_dataframe(bias_df[plot_sets], colors=None, frame='polygon', 
    legend_loc=(0.75, .95), fontsize=FONTSIZE, fontsize_features=FONTSIZE_SMALL,
    save_filename=FIG_RADAR_FNAME_FORMAT.format('all'), varlabels=FEATURE_NAMES_DICT)
radar_chart.plot_radar_from_dataframe(bias_df_tv[plot_sets], colors=None, frame='polygon',
    legend_loc=(0.75, .95), fontsize=FONTSIZE, fontsize_features=FONTSIZE_SMALL,
    save_filename=FIG_RADAR_FNAME_FORMAT.format('all_tv'), varlabels=FEATURE_NAMES_DICT)
radar_chart.plot_radar_from_dataframe(bias_df_max[plot_sets], colors=None, frame='polygon',
    legend_loc=(0.75, .95), fontsize=FONTSIZE, fontsize_features=FONTSIZE_SMALL,
    save_filename=FIG_RADAR_FNAME_FORMAT.format('all_max'), varlabels=FEATURE_NAMES_DICT)


## plot detailed distibutions
network_sets_dict_plots = {'All ASes': network_sets_dict['all'],
                        'RIPE Atlas': network_sets_dict['RIPE Atlas'], 
                        'RIPE RIS': network_sets_dict['RIPE RIS'], 
                        'RouteViews': network_sets_dict['RouteViews']}
# print_df.index = [n.replace('\n','') for n in FEATURE_NAMES_DICT.values()]                        
for k, d in network_sets_dict_plots.items():
    network_sets_dict_plots[k]  = d.replace('Educational/Research', 'Edu/Research')
# gdp.plot_all(network_sets_dict_plots, SAVE_PLOTS_DISTRIBUTION_FNAME_FORMAT, save_json=False, show_plot=False)