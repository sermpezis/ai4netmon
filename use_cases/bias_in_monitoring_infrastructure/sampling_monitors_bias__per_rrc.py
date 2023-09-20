import numpy as np
import pandas as pd
import json
import random
from matplotlib import pyplot as plt
from ai4netmon.Analysis.bias import bias_utils as bu
import os


from ai4netmon.Analysis.aggregate_data import data_collectors as dc
from collections import defaultdict
from ai4netmon.Analysis.bias import radar_chart



# load RRC 2 ASN data 
ris_peer_ip2asn, ris_peer_ip2rrc = dc.get_ripe_ris_data()

rrc2asn_dict = defaultdict(list)
for ip, rrc in ris_peer_ip2rrc.items():
    rrc2asn_dict[rrc].append( ris_peer_ip2asn[ip] )



## datasets
AGGREGATE_DATA_FNAME = '../../data/aggregate_data/asn_aggregate_data.csv'
BIAS_CSV_FNAME = './data/bias_values_per_rrc.csv'
BIAS_CSV_FNAME_NO_STUBS = './data/bias_values_per_rrc__no_stubs.csv'
OMIT_STUBS = False
if OMIT_STUBS:
    BIAS_CSV_FNAME = BIAS_CSV_FNAME_NO_STUBS
NB_SAMPLES = [10, 20, 50, 100, 200, 500, 1000]
NB_ITERATIONS = 100


# select features for visualization
FEATURE_NAMES_DICT = bu.get_features_dict_for_visualizations() 
FEATURES = list(FEATURE_NAMES_DICT.keys())


## load data
df = pd.read_csv(AGGREGATE_DATA_FNAME, header=0, index_col=0)
df['is_personal_AS'].fillna(0, inplace=True)
if OMIT_STUBS:
    df = df[df['AS_rel_degree']>1]

## calculate bias for all features
# define sets of interest
df_ris = df.loc[(df['is_ris_peer_v4']>0) | (df['is_ris_peer_v6']>0)]
ris_asns = list(df_ris.index)

network_sets_dict = dict()
network_sets_dict['all'] = df
network_sets_dict['RIPE RIS (all)'] = df_ris
for rrc, rrc_asns in rrc2asn_dict.items():
    network_sets_dict[rrc] = df.loc[set(rrc_asns)]

network_sets_dict_for_bias = {k:v[FEATURES] for k,v in network_sets_dict.items() if k != 'all'}

params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}
bias_df = bu.bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **params)

# print biases & save to csv
print('Bias per monitor set (columns) and per feature (rows)')
print_df = bias_df.copy()
print_df.index = [n.replace('\n','') for n in FEATURE_NAMES_DICT.values()]
print(print_df.round(2))
print_df.round(4).to_csv(BIAS_CSV_FNAME, header=True, index=True)



# print avg bias for infrastucture
print('### Avg bias for infrastructure ###')
print('RIPE RIS (all): '+ str(round(bias_df['RIPE RIS (all)'].mean(),2)))
for k,v in rrc2asn_dict.items():
    print('{}: {}({})\t {}'.format(k, len(v), len(set(v)), round(bias_df[k].mean(),2)))




#plotting 
FIG_SCATTER_SAVE_FNAME = './figures/Fig_scatter_bias_vs_sampling_per_rrc.png'
FIG_RADAR_SAVE_FNAME_FORMAT = './figures/Fig_radar_bias__per_rrc_{}.png'
FONTSIZE = 15
FONTSIZE_SMALL = 13

plt.figure(1)
list_of_rrcs = list(rrc2asn_dict.keys())
multihop_rrcs = ['rrc00', 'rrc24', 'rrc25']
non_multihop_rrcs = [r for r in list_of_rrcs if r not in multihop_rrcs]
# peers_per_rrc = [len(set(rrc2asn_dict[rrc])) for rrc in list_of_rrcs]
# bias_per_rrc = [bias_df[rrc].mean() for rrc in list_of_rrcs]
# plt.scatter(peers_per_rrc, bias_per_rrc, label='RRCs')
peers_per_rrc = [len(set(rrc2asn_dict[rrc])) for rrc in list_of_rrcs]
bias_per_rrc = [bias_df[rrc].mean() for rrc in list_of_rrcs]
plt.scatter([len(set(rrc2asn_dict[rrc])) for rrc in non_multihop_rrcs], [bias_df[rrc].mean() for rrc in non_multihop_rrcs], label='non-multihop RRCs', c='b')
plt.scatter([len(set(rrc2asn_dict[rrc])) for rrc in multihop_rrcs], [bias_df[rrc].mean() for rrc in multihop_rrcs], label='multihop RRCs', c='g')
for i,t in enumerate(list_of_rrcs):
    pos = (peers_per_rrc[i], bias_per_rrc[i])
    if t == 'rrc01':
        pos = (peers_per_rrc[i]-5, bias_per_rrc[i])
    elif t == 'rrc10':
        pos = (peers_per_rrc[i], bias_per_rrc[i]-0.02)
    elif t == 'rrc20':
        pos = (peers_per_rrc[i]-5, bias_per_rrc[i]-0.02)
    plt.annotate(t[3:], pos, fontsize=FONTSIZE_SMALL)
plt.axhline(y=bias_df['RIPE RIS (all)'].mean(), color='r', linestyle='--', label='RIPE RIS (all)')
plt.grid(True)
plt.legend(fontsize=FONTSIZE)
plt.xlabel('Number of peering ASNs', fontsize=FONTSIZE)
plt.ylabel('Average bias', fontsize=FONTSIZE)
plt.xticks(fontsize=FONTSIZE)
plt.yticks(fontsize=FONTSIZE)
plt.subplots_adjust(left=0.15, bottom=0.15)
plt.savefig(FIG_SCATTER_SAVE_FNAME)
plt.close()



plt.figure(2)
# 
bias_df['RIPE RIS (avg)'] = bias_df[list_of_rrcs].mean(axis=1)
bias_df['RIPE RIS (avg) multihop'] = bias_df[multihop_rrcs].mean(axis=1)
bias_df['RIPE RIS (avg) non multihop'] = bias_df[non_multihop_rrcs].mean(axis=1)
# radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', cmap='turbo', legend_loc=(0.9, .65),
    # save_filename=FIG_RADAR_SAVE_FNAME, varlabels=FEATURE_NAMES_DICT)
plot_df = bias_df[['RIPE RIS (all)']+list_of_rrcs]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', cmap='turbo', legend_loc=(0.9, .65),
    save_filename=FIG_RADAR_SAVE_FNAME_FORMAT.format('all'), varlabels=FEATURE_NAMES_DICT)
plot_df = bias_df[['RIPE RIS (all)', 'RIPE RIS (avg)']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', 
    save_filename=FIG_RADAR_SAVE_FNAME_FORMAT.format('all_vs_avg'), varlabels=FEATURE_NAMES_DICT)
plot_df = bias_df[['RIPE RIS (avg) multihop', 'RIPE RIS (avg) non multihop']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', legend_loc=(0.8, .95), 
    save_filename=FIG_RADAR_SAVE_FNAME_FORMAT.format('avg_multihop'), varlabels=FEATURE_NAMES_DICT)
