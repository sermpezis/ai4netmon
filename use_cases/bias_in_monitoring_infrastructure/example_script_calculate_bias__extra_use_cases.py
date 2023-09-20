import numpy as np
import pandas as pd
import json
import random
from matplotlib import pyplot as plt
from ai4netmon.Analysis.bias import bias_utils as bu
from ai4netmon.Analysis.bias import radar_chart
from collections import defaultdict



## datasets
AGGREGATE_DATA_FNAME = '../../data/aggregate_data/asn_aggregate_data.csv'
FIG_SAVENAME_FORMAT = './figures/fig_radar_{}.png'
BIAS_CSV_FNAME = './data/bias_values_ris_atlas_rv.csv'
ASN2ASN_DIST_FNAME = '../../data/misc/asn2asn__only_peers_pfx.json'
ORDERED_LIST_BIASES = 'https://raw.githubusercontent.com/sermpezis/ai4netmon/dev/TEMP_pavlos/bias_sort_nonRIS_asns/data/sorted_asns_by_ascending_biases.json'


# select features for visualization
FEATURE_NAMES_DICT = bu.get_features_dict_for_visualizations() 
FEATURES = list(FEATURE_NAMES_DICT.keys())

## load data
df = pd.read_csv(AGGREGATE_DATA_FNAME, header=0, index_col=0)
df['is_personal_AS'].fillna(0, inplace=True)


# load proximity victionary {originAS1: {peer1:proximity11, peer2: proximity12, ...}, originAS2: ...}
with open(ASN2ASN_DIST_FNAME, 'r') as f:
    asn2asn = json.load(f)

# find full feeding peers
feed = defaultdict(lambda : 0)
for o_asn, dict_o_asn in asn2asn.items():
    for m_ip, dist in dict_o_asn.items():
        feed[m_ip] +=1
full_feeders_ips = [m_ip for m_ip, nb_feeds in feed.items() if nb_feeds > 65000]
full_feeders_asns = [m_asn for m_ip, m_asn in ris_dict.items() if m_ip in full_feeders_ips]
ris_asns_full = [i for i in full_feeders_asns if i in df.index]



## calculate bias for all features
network_sets_dict = dict()
network_sets_dict['all'] = df
network_sets_dict['RIPE RIS (all)'] = df.loc[(df['is_ris_peer_v4']>0) | (df['is_ris_peer_v6']>0)]
network_sets_dict['RouteViews (all)'] = df.loc[df['is_routeviews_peer']>0]
network_sets_dict['RIPE RIS (full)'] = df.loc[ris_asns_full]

ris_asns = list(network_sets_dict['RIPE RIS (all)'].index)
order_bias = pd.read_json(ORDERED_LIST_BIASES, orient='records')['total'].to_list()
network_sets_dict['RIPE RIS +10'] = df.loc[ris_asns+order_bias[0:10]]
network_sets_dict['RIPE RIS +50'] = df.loc[ris_asns+order_bias[0:50]]
network_sets_dict['RIPE RIS +100'] = df.loc[ris_asns+order_bias[0:100]]
network_sets_dict['RIPE RIS +200'] = df.loc[ris_asns+order_bias[0:200]]
network_sets_dict['+10'] = df.loc[order_bias[0:10]]
network_sets_dict['+50'] = df.loc[order_bias[0:50]]
network_sets_dict['+100'] = df.loc[order_bias[0:100]]
network_sets_dict['+200'] = df.loc[order_bias[0:200]]



# calculate bias dataframes
network_sets_dict_for_bias = {k:v[FEATURES] for k,v in network_sets_dict.items() if k != 'all'}

params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}
bias_df = bu.bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **params)

params={'method':'total_variation', 'bins':10}
bias_df_tv = bu.bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **params)

params={'method':'max_variation', 'bins':10}
bias_df_max = bu.bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **params)



print('Bias per monitor set (columns) and per feature (rows)')
print_df = bias_df[['RIPE RIS (all)','RIPE RIS (full)', 'RouteViews (all)', '+10', '+50', '+100', '+200']].copy()
print_df.index = [n.replace('\n','') for n in FEATURE_NAMES_DICT.values()]
# print results
print(print_df.round(2))
# save results to file
# print_df.round(4).to_csv(BIAS_CSV_FNAME, header=True, index=True)


plot_df = bias_df[['RIPE RIS (all)','RIPE RIS (full)']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', save_filename=FIG_SAVENAME_FORMAT.format('RIPE_full'), varlabels=FEATURE_NAMES_DICT)


plot_df = bias_df[['RIPE RIS (all)','RIPE RIS +10','RIPE RIS +50','RIPE RIS +100', 'RIPE RIS +200']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', save_filename=FIG_SAVENAME_FORMAT.format('RIPE_plus'), varlabels=FEATURE_NAMES_DICT)


plot_df = bias_df[['RIPE RIS (all)','+10','+50','+100', '+200']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', save_filename=FIG_SAVENAME_FORMAT.format('RIPE_plus_only'), varlabels=FEATURE_NAMES_DICT)


print(bias_df[['RIPE RIS (all)','RIPE RIS +10', 'RIPE RIS +50', 'RIPE RIS +100', 'RIPE RIS +200', '+10', '+50', '+100', '+200']].sum(axis=0).round(2))
