import numpy as np
import pandas as pd
import json
import random
from matplotlib import pyplot as plt
from ai4netmon.Analysis.bias import bias_utils as bu
from ai4netmon.Analysis.bias import radar_chart
from ai4netmon.Analysis.bias import generate_distribution_plots as gdp
from ai4netmon.Analysis.aggregate_data import data_aggregation_tools as dat



## datasets
AGGREGATE_DATA_FNAME = '../../data/aggregate_data/asn_aggregate_data_20221128.csv'
FIG_RADAR_SAVENAME_FORMAT = './figures/fig_radar_{}.png'
BIAS_CSV_FNAME = './data/bias_values_ris_atlas_rv.csv'
FIG_RADAR_SAVENAME_FORMAT_NO_STUBS = './figures/fig_radar_{}__no_stubs.png'
BIAS_CSV_FNAME_NO_STUBS = './data/bias_values_ris_atlas_rv__no_stubs.csv'
SAVE_PLOTS_DISTRIBUTION_FNAME_FORMAT = './figures/Fig_{}_{}'
OMIT_STUBS = False


# select features for visualization
FEATURE_NAMES_DICT = bu.get_features_dict_for_visualizations() 
FEATURES = list(FEATURE_NAMES_DICT.keys())


## load data
# df = pd.read_csv(AGGREGATE_DATA_FNAME, header=0, index_col=0)
# df['is_personal_AS'].fillna(0, inplace=True)
df = dat.load_aggregated_dataframe(preprocess=True)

if OMIT_STUBS:
    df = df[df['AS_rel_degree']>1]
    FIG_RADAR_SAVENAME_FORMAT = FIG_RADAR_SAVENAME_FORMAT_NO_STUBS
    BIAS_CSV_FNAME = BIAS_CSV_FNAME_NO_STUBS

## calculate bias for all features

# define sets of interest
network_sets_dict = dict()
network_sets_dict['all'] = df
network_sets_dict['RIPE RIS (all)'] = df.loc[(df['is_ris_peer_v4']>0) | (df['is_ris_peer_v6']>0)]
network_sets_dict['RIPE RIS (v4)'] = df.loc[df['is_ris_peer_v4']>0]
network_sets_dict['RIPE RIS (v6)'] = df.loc[df['is_ris_peer_v6']>0]
network_sets_dict['RIPE Atlas (all)'] = df.loc[(df['nb_atlas_probes_v4']>0) | (df['nb_atlas_probes_v6']>0)]
network_sets_dict['RIPE Atlas (v4)'] = df.loc[df['nb_atlas_probes_v4']>0]
network_sets_dict['RIPE Atlas (v6)'] = df.loc[df['nb_atlas_probes_v6']>0]
network_sets_dict['RouteViews (all)'] = df.loc[df['is_routeviews_peer']>0]
network_sets_dict['bgptools (all)'] = df.loc[(df['is_bgptools_peer_v4']>0) | (df['is_bgptools_peer_v6']>0)]
network_sets_dict['bgptools (v4)'] = df.loc[(df['is_bgptools_peer_v4']>0)]
network_sets_dict['bgptools (v6)'] = df.loc[(df['is_bgptools_peer_v6']>0)]
network_sets_dict['RIPE RIS + RouteViews (all)'] = df.loc[(df['is_ris_peer_v4']>0) | (df['is_ris_peer_v6']>0) | (df['is_routeviews_peer']>0)]


# calculate bias dataframes
network_sets_dict_for_bias = {k:v[FEATURES] for k,v in network_sets_dict.items() if k != 'all'}

params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}
bias_df = bu.bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **params)

params={'method':'total_variation', 'bins':10}
bias_df_tv = bu.bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **params)

params={'method':'max_variation', 'bins':10}
bias_df_max = bu.bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **params)


# print biases & save to csv
print('Bias per monitor set (columns) and per feature (rows)')
print_df = bias_df[['RIPE RIS (all)','RIPE Atlas (all)', 'RouteViews (all)', 'RIPE RIS + RouteViews (all)', 'bgptools (all)']].copy()
print_df.index = [n.replace('\n','') for n in FEATURE_NAMES_DICT.values()]
print(print_df.round(2))
print_df.round(4).to_csv(BIAS_CSV_FNAME, header=True, index=True)


## plot the radar plot of biases
# all RIPE - details
plot_df = bias_df[['RIPE RIS (all)', 'RIPE RIS (v4)', 'RIPE RIS (v6)', 'RIPE Atlas (all)', 'RIPE Atlas (v4)', 'RIPE Atlas (v6)']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', save_filename=FIG_RADAR_SAVENAME_FORMAT.format('RIPE_detailed'), varlabels=FEATURE_NAMES_DICT)
# all RIPE
plot_df = bias_df[['RIPE RIS (all)','RIPE Atlas (all)']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', save_filename=FIG_RADAR_SAVENAME_FORMAT.format('RIPE'), varlabels=FEATURE_NAMES_DICT)
# all RIPE - TV
plot_df = bias_df_tv[['RIPE RIS (all)','RIPE Atlas (all)']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', save_filename=FIG_RADAR_SAVENAME_FORMAT.format('RIPE_tv'), varlabels=FEATURE_NAMES_DICT)
# all RIPE - Max
plot_df = bias_df_max[['RIPE RIS (all)','RIPE Atlas (all)']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', save_filename=FIG_RADAR_SAVENAME_FORMAT.format('RIPE_max'), varlabels=FEATURE_NAMES_DICT)
# all RIPE + RouteViews
plot_df = bias_df[['RIPE RIS (all)','RouteViews (all)']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', save_filename=FIG_RADAR_SAVENAME_FORMAT.format('RIPE_RV'), varlabels=FEATURE_NAMES_DICT)
plot_df = bias_df_tv[['RIPE RIS (all)','RouteViews (all)']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', save_filename=FIG_RADAR_SAVENAME_FORMAT.format('RIPE_RV_tv'), varlabels=FEATURE_NAMES_DICT)
# all bgptools, v4, v6
plot_df = bias_df[['bgptools (all)', 'bgptools (v4)', 'bgptools (v6)']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', save_filename=FIG_RADAR_SAVENAME_FORMAT.format('bgptools'), varlabels=FEATURE_NAMES_DICT)
# bgptools v4 vs RipeRIS v4
plot_df = bias_df[['bgptools (v4)','RIPE RIS (v4)']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', save_filename=FIG_RADAR_SAVENAME_FORMAT.format('bgptoolsv4_RIPERISv4'), varlabels=FEATURE_NAMES_DICT)
plot_df = bias_df[['bgptools (v6)','RIPE RIS (v6)']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', save_filename=FIG_RADAR_SAVENAME_FORMAT.format('bgptoolsv6_RIPERISv6'), varlabels=FEATURE_NAMES_DICT)
# bgptools vs RipeRIS+Routeviews vs RipeRIS vs Routeviews
plot_df = bias_df[['RIPE RIS (all)', 'RouteViews (all)', 'RIPE RIS + RouteViews (all)', 'bgptools (all)']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', save_filename=FIG_RADAR_SAVENAME_FORMAT.format('RIPERIS_RV_RIPERIS+RV_bgptools'), varlabels=FEATURE_NAMES_DICT)



## plot detailed distibutions
network_sets_dict_plots = {'All ASes': network_sets_dict['all'], 
                     'RIPE Atlas': network_sets_dict['RIPE Atlas (all)'], 
                     'RIPE RIS': network_sets_dict['RIPE RIS (all)'], 
                     'RouteViews': network_sets_dict['RouteViews (all)'],
                    'RIPE RIS + RouteViews': network_sets_dict['RIPE RIS + RouteViews (all)'],
                    'BGPtools': network_sets_dict['bgptools (all)']}
gdp.plot_all(network_sets_dict_plots, SAVE_PLOTS_DISTRIBUTION_FNAME_FORMAT, save_json=False, show_plot=False)
