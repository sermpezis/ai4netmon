import numpy as np
import pandas as pd
import json
import random
from matplotlib import pyplot as plt
from ai4netmon.Analysis.bias import bias_utils as bu
from ai4netmon.Analysis.bias import radar_chart
from collections import defaultdict



## datasets
AGGREGATE_DATA_FNAME = '../../data/aggregate_data/asn_aggregate_data_20211201.csv'
RIPE_RIS_FNAME = '../../data/misc/RIPE_RIS_peers_ip2asn.json'
ROUTEVIEWS_FNAME = '../../data/misc/RouteViews_peers.json'
FIG_SAVENAME_FORMAT = './figures/fig_radar_{}_STANs.png'


FEATURE_NAMES_DICT = {
    # Location-related info
    'AS_rank_source': 'RIR region',
    'AS_rank_iso': 'Location\n (country)',
    'AS_rank_continent': 'Location\n (continent)',
    # network size info
    'AS_rank_numberAsns': 'Customer cone\n (#ASNs)', 
    'AS_rank_numberPrefixes': 'Customer cone\n (#prefixes)',
    'AS_rank_numberAddresses': 'Customer cone\n (#addresses)',
    'AS_hegemony': 'AS hegemony',
    # Topology info
    'AS_rank_total': '#neighbors\n (total)',
    'AS_rank_peer': '#neighbors\n (peers)', 
    'AS_rank_customer': '#neighbors\n (customers)', 
    'AS_rank_provider': '#neighbors\n (providers)',
    # IXP related
    'peeringDB_ix_count': '#IXPs\n (PeeringDB)', 
    'peeringDB_fac_count': '#facilities\n (PeeringDB)', 
    'peeringDB_policy_general': 'Peering policy\n (PeeringDB)',
    # Network type
    'peeringDB_info_type': 'Network type\n (PeeringDB)',
    'peeringDB_info_ratio': 'Traffic ratio\n (PeeringDB)',
    'peeringDB_info_traffic': 'Traffic volume\n (PeeringDB)', 
    'peeringDB_info_scope': 'Scope\n (PeeringDB)',
    'is_personal_AS': 'Personal ASN', 
}
FEATURES = list(FEATURE_NAMES_DICT.keys())



## load data
df = pd.read_csv(AGGREGATE_DATA_FNAME, header=0, index_col=0)
df['is_personal_AS'].fillna(0, inplace=True)

# STAN_COUNTRIES = ['KZ', 'TJ', 'UZ', 'KG', 'TM', 'AF', 'PK']
STAN_COUNTRIES = ['KZ', 'TJ', 'UZ', 'KG', 'TM']
df_stan = df[df['AS_rank_iso'].isin(STAN_COUNTRIES)]


## calculate and plot bias
params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}
network_sets_dict = {'stan': df_stan[FEATURES]}
bias_df = bu.bias_score_dataframe(df[FEATURES], network_sets_dict, **params)
radar_chart.plot_radar_from_dataframe(bias_df, colors=None, frame='polygon', save_filename=FIG_SAVENAME_FORMAT.format('RIPE_stan'), varlabels=FEATURE_NAMES_DICT, show=True)




## plot detailed distibutions
from generate_distribution_plots import generate_plot_json, CDF_features, Histogram_features, plot_cdf, plot_histogram

SAVE_PLOTS_FNAME_FORMAT = './figures/Fig_{}_{}_STANs'
SHOW_PLOTS = False
dict_network_sets = dict()
dict_network_sets['all'] = df
dict_network_sets['stan'] = df_stan

for feature in CDF_features:
    data = generate_plot_json(feature, 'CDF', dict_network_sets)
    filename_no_ext = SAVE_PLOTS_FNAME_FORMAT.format('CDF',feature)
    plot_cdf(data, dict_network_sets, filename_no_ext, show_plot=SHOW_PLOTS)
for feature in Histogram_features:
    data = generate_plot_json(feature, 'histogram', dict_network_sets)
    filename_no_ext = SAVE_PLOTS_FNAME_FORMAT.format('Histogram',feature)
    plot_histogram(data, dict_network_sets, filename_no_ext, show_plot=SHOW_PLOTS)
