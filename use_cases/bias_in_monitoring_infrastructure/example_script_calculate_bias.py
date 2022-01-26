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
ROUTEVIEWS_FNAME = '../../data/misc/RouteViews_peers.json'
FIG_SAVENAME_FORMAT = './figures/fig_radar_{}.png'
BIAS_CSV_FNAME = './data/bias_values_ris_atlas_rv.csv'

## features
CATEGORICAL_FEATURES =  ['AS_rank_source', 'AS_rank_iso', 'AS_rank_continent', 'is_personal_AS', 'peeringDB_info_ratio', 
'peeringDB_info_traffic', 'peeringDB_info_scope', 'peeringDB_info_type', 'peeringDB_policy_general']

NUMERICAL_FEATURES =  ['AS_rank_numberAsns', 'AS_rank_numberPrefixes', 'AS_rank_numberAddresses', 'AS_rank_total',
'AS_rank_customer', 'AS_rank_peer', 'AS_rank_provider', 'peeringDB_ix_count', 'peeringDB_fac_count', 'AS_hegemony']

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
ris_asns_v4 = [i for k,i in ris_dict.items() if (':' not in k) and (i in df.index)]
ris_asns_v6 = [i for k,i in ris_dict.items() if (':' in k) and (i in df.index)]

with open(ROUTEVIEWS_FNAME, 'r') as f:
    routeviews_asns = json.load(f)
routeviews_asns = [i for i in routeviews_asns if i in df.index]

## calculate bias for all features
bias_df = pd.DataFrame(index=FEATURES)
bias_df_tv = pd.DataFrame(index=FEATURES)
bias_df_max = pd.DataFrame(index=FEATURES)

NETWORK_SETS = ['all', 
                'RIPE RIS (all)', 
                'RIPE RIS (v4)', 
                'RIPE RIS (v6)', 
                'RIPE Atlas (all)', 
                'RIPE Atlas (v4)', 
                'RIPE Atlas (v6)', 
                'RouteViews (all)',
                'RIPE RIS + RouteViews (all)']
network_sets_dict = dict()
network_sets_dict['all'] = df
network_sets_dict['RIPE RIS (all)'] = df.loc[ris_asns]
network_sets_dict['RIPE RIS (v4)'] = df.loc[ris_asns_v4]
network_sets_dict['RIPE RIS (v6)'] = df.loc[ris_asns_v6]
network_sets_dict['RIPE Atlas (all)'] = df.loc[ (df['nb_atlas_probes_v4'] >0) | (df['nb_atlas_probes_v6'] >0) ]
network_sets_dict['RIPE Atlas (v4)'] = df.loc[df['nb_atlas_probes_v4'] >0]
network_sets_dict['RIPE Atlas (v6)'] = df.loc[df['nb_atlas_probes_v6'] >0]
network_sets_dict['RouteViews (all)'] = df.loc[routeviews_asns]
network_sets_dict['RIPE RIS + RouteViews (all)'] = df.loc[ris_asns+routeviews_asns]



for feature in FEATURES:
	params={'data_type':get_feature_type(feature), 'bins':10, 'alpha':0.01}
	network_data_processed = dict()
	for s in NETWORK_SETS:
		d = network_sets_dict[s][feature].copy()
		d = d[(d.notnull())]
		if params['data_type'] == 'numerical': # pre-processing for the numerical cases
			# d[d<=1] = 0.9#np.nan
			d = np.log(d)
			d[np.isinf(d)] = -0.1
		network_data_processed[s] = d

	for s in NETWORK_SETS[1:]:
		bias_df.loc[feature,s] = bu.bias_score(network_data_processed['all'], network_data_processed[s], method='kl_divergence', **params)
		bias_df_tv.loc[feature,s] = bu.bias_score(network_data_processed['all'], network_data_processed[s], method='total_variation', **params)
		bias_df_max.loc[feature,s] = bu.bias_score(network_data_processed['all'], network_data_processed[s], method='max_variation', **params)

print('Bias per monitor set (columns) and per feature (rows)')
print_df = bias_df[['RIPE RIS (all)','RIPE Atlas (all)', 'RouteViews (all)']].copy()
print_df.index = [n.replace('\n','') for n in FEATURE_NAMES_DICT.values()]
# print results
print(print_df.round(2))
# save results to file
print_df.round(4).to_csv(BIAS_CSV_FNAME, header=True, index=True)


## plot the radar plot of biases
# all RIPE - details
plot_df = bias_df[['RIPE RIS (all)', 'RIPE RIS (v4)', 'RIPE RIS (v6)', 'RIPE Atlas (all)', 'RIPE Atlas (v4)', 'RIPE Atlas (v6)']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', save_filename=FIG_SAVENAME_FORMAT.format('RIPE_detailed'), varlabels=FEATURE_NAMES_DICT)
# all RIPE
plot_df = bias_df[['RIPE RIS (all)','RIPE Atlas (all)']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', save_filename=FIG_SAVENAME_FORMAT.format('RIPE'), varlabels=FEATURE_NAMES_DICT)
# all RIPE - TV
plot_df = bias_df_tv[['RIPE RIS (all)','RIPE Atlas (all)']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', save_filename=FIG_SAVENAME_FORMAT.format('RIPE_tv'), varlabels=FEATURE_NAMES_DICT)
# all RIPE - Max
plot_df = bias_df_max[['RIPE RIS (all)','RIPE Atlas (all)']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', save_filename=FIG_SAVENAME_FORMAT.format('RIPE_max'), varlabels=FEATURE_NAMES_DICT)
# all RIPE + RouteViews
plot_df = bias_df[['RIPE RIS (all)','RouteViews (all)']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', save_filename=FIG_SAVENAME_FORMAT.format('RIPE_RV'), varlabels=FEATURE_NAMES_DICT)
plot_df = bias_df_tv[['RIPE RIS (all)','RouteViews (all)']]
radar_chart.plot_radar_from_dataframe(plot_df, colors=None, frame='polygon', save_filename=FIG_SAVENAME_FORMAT.format('RIPE_RV_tv'), varlabels=FEATURE_NAMES_DICT)