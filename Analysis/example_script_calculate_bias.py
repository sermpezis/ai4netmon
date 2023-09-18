import numpy as np
import pandas as pd
import json
import random
from matplotlib import pyplot as plt
from ai4netmon.Analysis.bias import bias_utils as bu
from ai4netmon.Analysis.bias import radar_chart




### Example 1 - simple example
print('####### Example 1 - simple example of calculating bias')
v1 = np.array([1,2,3,4,5,6,7,8,9])
v2 = np.array([1,3,9])
print('Target data (groundtruth):', list(v1))
print('Sample data:', list(v1))
print('Bias score (KL - numerical): {}'.format(bu.bias_score(v1, v2, method='kl_divergence', **{'data_type':'numerical', 'bins':3, 'alpha':0.01})))
print('Bias score (KL - categorical): {}'.format(bu.bias_score(v1, v2, method='kl_divergence', **{'data_type':'categorical', 'alpha':0.01})))
print('Bias score (KS-test): {}'.format(bu.bias_score(v1,v2, method='ks_test')))
print()



### Example 2 - bias in RIPE monitors
print('####### Example 2 - bias in RIPE monitors')

## datasets
AGGREGATE_DATA_FNAME = '../data/aggregate_data/asn_aggregate_data.csv'
# RIPE_RIS_FNAME = '../data/misc/RIPE_RIS_peers_ip2asn.json'

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

# with open(RIPE_RIS_FNAME, 'r') as f:
# 	ris_dict = json.load(f)
# ris_asns = [i for i in ris_dict.values() if i in df.index]
# ris_asns_v4 = [i for k,i in ris_dict.items() if (':' not in k) and (i in df.index)]
# ris_asns_v6 = [i for k,i in ris_dict.items() if (':' in k) and (i in df.index)]


## calculate bias for all features
bias_df = pd.DataFrame(index=FEATURES)

network_sets = ['all', 'RIPE RIS', 'RIPE Atlas', 'sample500', 'sample1000']
network_sets_dict = dict()
network_sets_dict['all'] = df
# network_sets_dict['RIPE RIS'] = df.loc[ris_asns]
network_sets_dict['RIPE RIS'] = df.loc[(df['is_ris_peer_v4']>0) | (df['is_ris_peer_v6']>0)]
network_sets_dict['RIPE Atlas'] = df.loc[ (df['nb_atlas_probes_v4'] >0) | (df['nb_atlas_probes_v6'] >0) ]
network_sets_dict['sample500'] = df.loc[random.sample(list(df.index), 500)]
network_sets_dict['sample1000'] = df.loc[random.sample(list(df.index), 1000)]

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
print(bias_df.round(2))


## plot the radar plot of biases
plot_df = bias_df[['RIPE RIS', 'RIPE Atlas']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', save_filename='fig_radar.png', show=True)
