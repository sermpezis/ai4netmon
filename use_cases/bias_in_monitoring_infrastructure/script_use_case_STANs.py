import numpy as np
import pandas as pd
import json
import random
from matplotlib import pyplot as plt
from ai4netmon.Analysis.bias import bias_utils as bu
from ai4netmon.Analysis.bias import radar_chart
from ai4netmon.Analysis.bias import generate_distribution_plots as gdp


## datasets
AGGREGATE_DATA_FNAME = '../../data/aggregate_data/asn_aggregate_data_20211201.csv'
FIG_RADAR_SAVENAME_FORMAT = './figures/fig_radar_{}_STANs.png'
SAVE_PLOTS_DISRIBUTION_FNAME_FORMAT = './figures/Fig_{}_{}_STANs'


FEATURE_NAMES_DICT = bu.get_features_dict_for_visualizations() 
FEATURES = list(FEATURE_NAMES_DICT.keys())

## load data
df = pd.read_csv(AGGREGATE_DATA_FNAME, header=0, index_col=0)
df['is_personal_AS'].fillna(0, inplace=True)

STAN_COUNTRIES = ['KZ', 'TJ', 'UZ', 'KG', 'TM']
df_stan = df[df['AS_rank_iso'].isin(STAN_COUNTRIES)]

## calculate and plot bias
params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}
network_sets_dict = {'stan': df_stan[FEATURES]}
bias_df = bu.bias_score_dataframe(df[FEATURES], network_sets_dict, **params)
radar_chart.plot_radar_from_dataframe(bias_df, colors=None, frame='polygon', save_filename=FIG_RADAR_SAVENAME_FORMAT.format('RIPE_stan'), varlabels=FEATURE_NAMES_DICT, show=True)


## plot detailed distibutions
dict_network_sets = dict()
network_sets_dict['all'] = df
dict_network_sets['stan'] = df_stan
gdp.plot_all(dict_network_sets, SAVE_PLOTS_DISRIBUTION_FNAME_FORMAT, save_json=False, show_plot=False)