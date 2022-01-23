import numpy as np
import pandas as pd
import json
import random
from matplotlib import pyplot as plt
from ai4netmon.Analysis.bias import bias_utils as bu
from ai4netmon.Analysis.bias import radar_chart

## datasets
AGGREGATE_DATA_FNAME = '../../data/aggregate_data/asn_aggregate_data_20211201.csv'
RIPE_RIS_FNAME = '../../data/misc/RIPE_RIS_peers_ip2asn.json'
SELECTED_FNAME = '../../use_cases/ripe_ris_subsampling/dataset_selected_monitors_ripe_ris_pathlens_100k_greedy_min_full_v4.json'
TIER1_FNAME = '../../data/misc/tier1_networks.json'

## features
CATEGORICAL_FEATURES =  ['AS_rank_source', 'AS_rank_iso', 'AS_rank_continent', 'is_personal_AS', 'peeringDB_info_ratio', 
'peeringDB_info_traffic', 'peeringDB_info_scope', 'peeringDB_info_type', 'peeringDB_policy_general']
NUMERICAL_FEATURES =  ['AS_rank_numberAsns', 'AS_rank_numberPrefixes', 'AS_rank_numberAddresses', 'AS_rank_total',
'AS_rank_customer', 'AS_rank_peer', 'AS_rank_provider','peeringDB_ix_count', 'peeringDB_fac_count', 'AS_hegemony']
# FEATURES = CATEGORICAL_FEATURES+NUMERICAL_FEATURES


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
non_ris_asns = list(set(df.index)-set(ris_asns))

with open(SELECTED_FNAME, 'r') as f:
	sel_list = json.load(f)
sel_asns = [ris_dict[i] for i in sel_list if ris_dict[i] in df.index][0:100]

with open(TIER1_FNAME, 'r') as f:
	tier1 = json.load(f)
tier1 = [i for i in tier1 if i in df.index]

## calculate bias for all features
bias_df = pd.DataFrame(index=FEATURES)
bias_df_tv = pd.DataFrame(index=FEATURES)
bias_df_max = pd.DataFrame(index=FEATURES)



network_sets = ['all', 'RIPE RIS']
network_sets_dict = dict()
network_sets_dict['all'] = df
network_sets_dict['RIPE RIS'] = df.loc[ris_asns]
network_sets_dict['selected'] = df.loc[sel_asns]

bias_diff = dict()

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

	for ASN in non_ris_asns:
		d = df.loc[ris_asns + [ASN]][feature].copy()
		d = d[(d.notnull())]
		if params['data_type'] == 'numerical': # pre-processing for the numerical cases
			# d[d<=1] = 0.9#np.nan
			d = np.log(d)
			d[np.isinf(d)] = -0.1
		bias_df.loc[feature,ASN] = bu.bias_score(network_data_processed['all'], d, method='kl_divergence', **params)

	for s in network_sets[1:]:
		bias_df.loc[feature,s] = bu.bias_score(network_data_processed['all'], network_data_processed[s], method='kl_divergence', **params)

bias_diff = bias_df.sum(axis=0)

bias_df.to_csv('./data/bias_df.csv', index=True, header=True)
bias_diff.sort_values().to_csv('./data/bias_total_df.csv', index=True, header=True)
# print(bias_diff.value_counts())

# sq=bias_diff.value_counts()
print(bias_diff.sort_values())


print(bias_diff.round(2))
