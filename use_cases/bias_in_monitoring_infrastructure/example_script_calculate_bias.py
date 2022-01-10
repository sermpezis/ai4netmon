import numpy as np
import pandas as pd
import json
import random
from matplotlib import pyplot as plt
from ai4netmon.Analysis.bias import bias_utils as bu
from ai4netmon.Analysis.bias import radar_chart



print('####### Example 2 - bias in RIPE monitors')

## datasets
AGGREGATE_DATA_FNAME = '../../data/aggregate_data/asn_aggregate_data_20211201.csv'
RIPE_RIS_FNAME = '../../data/misc/RIPE_RIS_peers_ip2asn.json'

## features
CATEGORICAL_FEATURES =  ['AS_rank_source', 'AS_rank_iso', 'AS_rank_continent', 'is_personal_AS', 'peeringDB_info_ratio', 
'peeringDB_info_traffic', 'peeringDB_info_scope', 'peeringDB_info_type', 'peeringDB_policy_general']

NUMERICAL_FEATURES =  ['AS_rank_numberAsns', 'AS_rank_numberPrefixes', 'AS_rank_numberAddresses', 'AS_rank_total',
'AS_rank_customer', 'AS_rank_peer', 'AS_rank_provider', 'peeringDB_info_prefixes4', 'peeringDB_info_prefixes6', 
'peeringDB_ix_count', 'peeringDB_fac_count', 'AS_hegemony']

FEATURES = CATEGORICAL_FEATURES+NUMERICAL_FEATURES


## useful methods
def get_feature_type(feature):
	if feature in CATEGORICAL_FEATURES:
		data_type = 'categorical'
	elif feature in NUMERICAL_FEATURES:
		data_type = 'numerical'
	else:
		raise ValueError
	return data_type



## load data
df = pd.read_csv(AGGREGATE_DATA_FNAME, header=0, index_col=0)
df['is_personal_AS'].fillna(0, inplace=True)

with open(RIPE_RIS_FNAME, 'r') as f:
	ris_dict = json.load(f)
ris_asns = [i for i in ris_dict.values() if i in df.index]
ris_asns_v4 = [i for k,i in ris_dict.items() if (':' not in k) and (i in df.index)]
ris_asns_v6 = [i for k,i in ris_dict.items() if (':' in k) and (i in df.index)]


## calculate bias for all features
bias_df = pd.DataFrame(index=FEATURES)

network_sets = ['all', 'RIPE RIS (all)', 'RIPE RIS (v4)', 'RIPE RIS (v6)', 'RIPE Atlas (all)', 'RIPE Atlas (v4)', 'RIPE Atlas (v6)']
network_sets_dict = dict()
network_sets_dict['all'] = df
network_sets_dict['RIPE RIS (all)'] = df.loc[ris_asns]
network_sets_dict['RIPE RIS (v4)'] = df.loc[ris_asns_v4]
network_sets_dict['RIPE RIS (v6)'] = df.loc[ris_asns_v6]
network_sets_dict['RIPE Atlas (all)'] = df.loc[ (df['nb_atlas_probes_v4'] >0) | (df['nb_atlas_probes_v6'] >0) ]
network_sets_dict['RIPE Atlas (v4)'] = df.loc[df['nb_atlas_probes_v4'] >0]
network_sets_dict['RIPE Atlas (v6)'] = df.loc[df['nb_atlas_probes_v6'] >0]



for feature in FEATURES:
	params={'data_type':get_feature_type(feature), 'bins':10, 'alpha':0.01}
	network_data_processed = dict()
	for s in network_sets:
		d = network_sets_dict[s][feature].copy()
		d = d[(d.notnull())]
		if params['data_type'] == 'numerical': # pre-processing for the numerical cases
			# d[d<=1] = 0.9#np.nan
			d = np.log(d)
			d[np.isinf(d)] = -0.1
		network_data_processed[s] = d

	for s in network_sets[1:]:
		bias_df.loc[feature,s] = bu.bias_score(network_data_processed['all'], network_data_processed[s], method='kl_divergence', **params)

print('Bias per monitor set (columns) and per feature (rows)')
print(bias_df[['RIPE RIS (all)','RIPE Atlas (all)']].round(2))


## plot the radar plot of biases
plot_df = bias_df[network_sets[1:]]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', save_filename='fig_radar_detailed.png')
plot_df = bias_df[['RIPE RIS (all)','RIPE Atlas (all)']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', save_filename='fig_radar.png')