import numpy as np
import pandas as pd
import json
import random
from matplotlib import pyplot as plt
from ai4netmon.Analysis.bias import bias_utils as bu
from ai4netmon.Analysis.aggregate_data import data_aggregation_tools as dat
import os
from ai4netmon.Analysis.aggregate_data import data_collectors as dc
from collections import defaultdict
from ai4netmon.Analysis.bias import radar_chart



## datasets
ROUTEVIEWS_FILE = '../../data/misc/RouteViews_20220402.txt'
BIAS_CSV_FNAME = './data/bias_values_per_rrc.csv'
FIG_SCATTER_SAVE_FNAME = './figures/Fig_scatter_bias_vs_sampling_per_rc_{}.png'
FIG_RADAR_SAVE_FNAME_FORMAT = './figures/Fig_radar_bias__per_rrc_{}.png'

# select features for visualization
FEATURE_NAMES_DICT = bu.get_features_dict_for_visualizations() 
FEATURES = list(FEATURE_NAMES_DICT.keys())




## load data
df = dat.load_aggregated_dataframe(preprocess=True)

# load RRC 2 ASN data 
ris_peer_ip2asn, ris_peer_ip2rrc = dc.get_ripe_ris_data()
rrc2asn_dict = defaultdict(list)
for ip, rrc in ris_peer_ip2rrc.items():
    rrc2asn_dict[rrc].append( ris_peer_ip2asn[ip] )

# load RouteViews RC 2 ASN data 
df_rv_rc = pd.read_csv(ROUTEVIEWS_FILE, delimiter='|')
rv_rc = list(set(df_rv_rc['ROUTEVIEWS_COLLECTOR']))
rv_rc2asn_dict = dict()
for rc in rv_rc:
    rv_rc2asn_dict[rc] = list(set(df_rv_rc[df_rv_rc['ROUTEVIEWS_COLLECTOR']==rc]['AS_NUMBER']))


## calculate bias for all features
# define sets of interest
df_ris = df.loc[(df['is_ris_peer_v4']>0) | (df['is_ris_peer_v6']>0)]
ris_asns = list(df_ris.index)
df_rv = df.loc[df['is_routeviews_peer']>0]
rv_asns = list(df_rv.index)

network_sets_dict = dict()
network_sets_dict['all'] = df
network_sets_dict['RIS'] = df_ris
network_sets_dict['RV'] = df_rv
for rrc, rrc_asns in rrc2asn_dict.items():
    network_sets_dict[rrc] = df.loc[[i for i in set(rrc_asns) if i in df.index]]
for rc, rc_asns in rv_rc2asn_dict.items():
    network_sets_dict[rc] = df.loc[[i for i in set(rc_asns) if i in df.index]]

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
print('RIS: '+ str(round(bias_df['RIS'].mean(),2)))
for k,v in rrc2asn_dict.items():
    print('{}: {}({})\t {}'.format(k, len(v), len(set(v)), round(bias_df[k].mean(),2)))
print()
print('RV: '+ str(round(bias_df['RV'].mean(),2)))
for k,v in rv_rc2asn_dict.items():
    print('{}: {}({})\t {}'.format(k, len(v), len(set(v)), round(bias_df[k].mean(),2)))


#plotting 

FONTSIZE = 15
FONTSIZE_SMALL = 13
MARKERS = {'RIS':'o', 'RV':'x'}
MARKERSIZE = 60

plt.figure(1)
def plot_bias_peers_per_rc(dict_rc_asn, multihop_rrcs, project, plot_per_project=False):
    list_of_rrcs = list(dict_rc_asn.keys())    
    non_multihop_rrcs = [r for r in list_of_rrcs if r not in multihop_rrcs]
    peers_per_rrc = [len(set(dict_rc_asn[rrc])) for rrc in list_of_rrcs]
    bias_per_rrc = [bias_df[rrc].mean() for rrc in list_of_rrcs]
    plt.scatter([len(set(dict_rc_asn[rrc])) for rrc in non_multihop_rrcs], [bias_df[rrc].mean() for rrc in non_multihop_rrcs], 
            label='{}: non-multihop RCs'.format(project), c='b', marker=MARKERS[project], s=MARKERSIZE)
    plt.scatter([len(set(dict_rc_asn[rrc])) for rrc in multihop_rrcs], [bias_df[rrc].mean() for rrc in multihop_rrcs], 
            label='{}: multihop RCs'.format(project), c='g', marker=MARKERS[project], s=MARKERSIZE)
    plt.grid(True)
    plt.legend(fontsize=FONTSIZE)
    plt.xlabel('number of vantage points', fontsize=FONTSIZE)
    plt.ylabel('avg. bias score', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    # plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.tight_layout(pad = 0.0)

    if plot_per_project:
        # for i,t in enumerate(list_of_rrcs):
        #     pos = (peers_per_rrc[i], bias_per_rrc[i])
        #     if t == 'rrc01':
        #         pos = (peers_per_rrc[i]-5, bias_per_rrc[i])
        #     elif t == 'rrc10':
        #         pos = (peers_per_rrc[i], bias_per_rrc[i]-0.02)
        #     elif t == 'rrc20':
        #         pos = (peers_per_rrc[i]-5, bias_per_rrc[i]-0.02)
        #     if project == 'RIS':
        #         plt.annotate(t[3:], pos, fontsize=FONTSIZE_SMALL)
        #     elif project == 'RV':
        #         if t.startswith('route-views.'):
        #             plt.annotate(t.split('.')[1], pos, fontsize=FONTSIZE_SMALL)
        #         else:
        #             if t.split('.')[0][-1]=='s':
        #                 plt.annotate('rv', pos, fontsize=FONTSIZE_SMALL)
        #             else:
        #                 plt.annotate('rv'+t.split('.')[0][-1], pos, fontsize=FONTSIZE_SMALL)
        # plt.axhline(y=bias_df[project].mean(), color='r', linestyle='--', label=project)
        plt.savefig(FIG_SCATTER_SAVE_FNAME.format(project))
        plt.close()
    else:
        plt.axis([-5,100,0,0.8])

        

multihop_rrcs = dict()
multihop_rrcs['RIS'] = ['rrc00', 'rrc24', 'rrc25']
multihop_rrcs['RV'] = ['route-views.sfmix.routeviews.org',
'route-views.chicago.routeviews.org',
'route-views.mwix.routeviews.org',
'route-views.eqix.routeviews.org',
'route-views.flix.routeviews.org',
'route-views.peru.routeviews.org',
'route-views.fortaleza.routeviews.org',
'route-views.chile.routeviews.org',
'route-views.rio.routeviews.org',
'route-views2.saopaulo.routeviews.org',
'route-views.gixa.routeviews.org',
'route-views.napafrica.routeviews.org',
'route-views.linx.routeviews.org',
'route-views.amsix.routeviews.org',
'route-views.siex.routeviews.org',
'route-views.uaeix.routeviews.org',
'route-views.bdix.routeviews.org',
'route-views.bknix.routeviews.org',
'route-views.phoix.routeviews.org',
'route-views.gorex.routeviews.org']
multihop_rrcs['RV'] = list( set(multihop_rrcs['RV']).intersection(set(rv_rc2asn_dict.keys())) )



plot_bias_peers_per_rc(rrc2asn_dict, multihop_rrcs['RIS'] ,'RIS' )
plot_bias_peers_per_rc(rv_rc2asn_dict, multihop_rrcs['RV'] , 'RV')
plt.savefig(FIG_SCATTER_SAVE_FNAME.format('ALL'))
plt.close()
plot_bias_peers_per_rc(rrc2asn_dict, multihop_rrcs['RIS'] ,'RIS', True)
plot_bias_peers_per_rc(rv_rc2asn_dict, multihop_rrcs['RV'] , 'RV', True)





# plt.figure(2)
# # 
# bias_df['RIPE RIS (avg)'] = bias_df[list_of_rrcs].mean(axis=1)
# bias_df['RIPE RIS (avg) multihop'] = bias_df[multihop_rrcs].mean(axis=1)
# bias_df['RIPE RIS (avg) non multihop'] = bias_df[non_multihop_rrcs].mean(axis=1)
# # radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', cmap='turbo', legend_loc=(0.9, .65),
#     # save_filename=FIG_RADAR_SAVE_FNAME, varlabels=FEATURE_NAMES_DICT)
# plot_df = bias_df[['RIPE RIS (all)']+list_of_rrcs]
# radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', cmap='turbo', legend_loc=(0.9, .65),
#     save_filename=FIG_RADAR_SAVE_FNAME_FORMAT.format('all'), varlabels=FEATURE_NAMES_DICT)
# plot_df = bias_df[['RIPE RIS (all)', 'RIPE RIS (avg)']]
# radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', 
#     save_filename=FIG_RADAR_SAVE_FNAME_FORMAT.format('all_vs_avg'), varlabels=FEATURE_NAMES_DICT)
# plot_df = bias_df[['RIPE RIS (avg) multihop', 'RIPE RIS (avg) non multihop']]
# radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', legend_loc=(0.8, .95), 
#     save_filename=FIG_RADAR_SAVE_FNAME_FORMAT.format('avg_multihop'), varlabels=FEATURE_NAMES_DICT)
