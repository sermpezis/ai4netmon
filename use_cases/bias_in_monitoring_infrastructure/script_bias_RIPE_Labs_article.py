import numpy as np
import pandas as pd
import json
import random
from matplotlib import pyplot as plt
from ai4netmon.Analysis.bias import bias_utils as bu
from ai4netmon.Analysis.bias import radar_chart
from ai4netmon.Analysis.aggregate_data import data_aggregation_tools as dat
import os


## datasets
FIG_RADAR_FNAME_FORMAT = './figures/Fig_RIPE_LABS_radar_{}_NEW.png'

# select features for visualization
FEATURE_NAMES_DICT = bu.get_features_dict_for_visualizations() 
FEATURES = list(FEATURE_NAMES_DICT.keys())


## load data
df = dat.load_aggregated_dataframe(preprocess=True)
df_ris = df.loc[(df['is_ris_peer_v4']>0) | (df['is_ris_peer_v6']>0)]
ris_asns = list(df_ris.index)
df_rv = df.loc[df['is_routeviews_peer']>0]
rv_asns = list(df_rv.index)
df_atlas = df.loc[(df['nb_atlas_probes_v4']>0) | (df['nb_atlas_probes_v6']>0)]
atlas_asns = list(df_atlas.index)

network_sets_dict = dict()
network_sets_dict['all'] = df
network_sets_dict['RIPE Atlas'] = df_atlas
network_sets_dict['RIPE RIS'] = df_ris
network_sets_dict['RouteViews'] = df_rv
network_sets_dict['RIS & RouteViews'] = df.loc[(df['is_ris_peer_v4']>0) | (df['is_ris_peer_v6']>0) | (df['is_routeviews_peer']>0)]


network_sets_dict_for_bias = {k:v[FEATURES] for k,v in network_sets_dict.items() if k != 'all'}

params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}
bias_df = bu.bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **params)

# print biases & save to csv
print('Bias per monitor set (columns) and per feature (rows)')
print_df = bias_df.copy()
print_df.index = [n.replace('\n','') for n in FEATURE_NAMES_DICT.values()]
print(print_df.round(2))
# print_df.round(4).to_csv(BIAS_CSV_FNAME, header=True, index=True)

    
plot_df = bias_df[['RIPE Atlas', 'RIPE RIS', 'RouteViews']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', save_filename=FIG_RADAR_FNAME_FORMAT.format('all'), varlabels=FEATURE_NAMES_DICT)
#
plot_df = bias_df[['RIPE Atlas', 'RIPE RIS']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', save_filename=FIG_RADAR_FNAME_FORMAT.format('RA_RIS'), varlabels=FEATURE_NAMES_DICT)
#
plot_df = bias_df[['RIPE RIS', 'RouteViews']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', save_filename=FIG_RADAR_FNAME_FORMAT.format('RIS_RV'), varlabels=FEATURE_NAMES_DICT)
#
plot_df = bias_df[['RIPE RIS', 'RouteViews', 'RIS & RouteViews']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', save_filename=FIG_RADAR_FNAME_FORMAT.format('RIS_and_RV'), varlabels=FEATURE_NAMES_DICT)
#


print([len(i) for i in [ris_asns, rv_asns, atlas_asns]])