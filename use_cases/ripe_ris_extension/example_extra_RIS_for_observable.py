import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from ai4netmon.Analysis.aggregate_data import data_aggregation_tools as dat
from ai4netmon.Analysis.bias import bias_utils as bu


BIAS_DF = './data/bias_df__no_stubs_RIS.csv'
BIAS_DF_GROUPS = './data/bias_df__no_stubs_RIS_groups.{}'
FIG_BIAS_SAVENAME = './figures/fig_cdf_bias_vs_nb_extra_monitors.png'
MAX_EXTRA = 400

FONTSIZE = 15
LINEWIDTH = 2
colors_infra = {'all':'k', 'RIS':'b', 'RV':'r', 'Atlas':'g', 'RIS & RV':'m'}


FEATURE_NAMES_DICT = bu.get_features_dict_for_visualizations() 
FEATURES = list(FEATURE_NAMES_DICT.keys())
FEATURE_NAMES_DICT_NO_NEWLINE = {k:v.replace('\n','') for k,v in FEATURE_NAMES_DICT.items()}


'''
Load AI4NetMon dataframe (with ASNs data)
'''
df = dat.load_aggregated_dataframe(preprocess=True)
df.index = df.index.astype('int')

# select columns to include
only_columns = ['AS_rank_source', 'AS_rank_numberAsns', 'AS_rank_numberPrefixes', 'AS_rank_iso', 'AS_rank_total', 
	'peeringDB_info_scope', 'peeringDB_info_type', 'peeringDB_policy_general', 'peeringDB_ix_count', 'peeringDB_fac_count',
	'is_in_peeringDB', 'nb_atlas_probes_v4', 'nb_atlas_probes_v6', 'is_routeviews_peer']
df = df[only_columns] 

# change column names
new_column_names = []
for i, cn in enumerate(df.columns):
	if cn in FEATURE_NAMES_DICT_NO_NEWLINE.keys():
		new_column_names.append(FEATURE_NAMES_DICT_NO_NEWLINE[cn])
	else:
		new_column_names.append(cn)
df.columns = new_column_names



'''
Load dataframe with bias scores
'''
bias_df = pd.read_csv(BIAS_DF, header=0, index_col=0).transpose()
bias_df.columns = [FEATURE_NAMES_DICT_NO_NEWLINE[i] for i in bias_df.columns]

# transform absolute bias scores to relative scores in 100%
bias_diff = (bias_df/bias_df.loc['RIPE RIS',:]-1)*100

# group biases per group of dimensions (to reduce data size)
FEATURE_GROUPS = {
        'Location': ['RIR region', 'Location (country)', 'Location (continent)'],
        'Network size': ['Customer cone (#ASNs)', 'Customer cone (#prefixes)', 'Customer cone (#addresses)', 'AS hegemony'],
        'Topology': ['#neighbors (total)', '#neighbors (peers)', '#neighbors (customers)', '#neighbors (providers)'],
        'IXP related': ['#IXPs (PeeringDB)', '#facilities (PeeringDB)', 'Peering policy (PeeringDB)'],
        'Network type': ['Network type (PeeringDB)', 'Traffic ratio (PeeringDB)', 'Traffic volume (PeeringDB)', 'Scope (PeeringDB)', 'Personal ASN']
        }
new_bias_diff = pd.DataFrame()
for k,v in FEATURE_GROUPS.items():
	new_bias_diff[k+' bias'] = bias_diff[v].mean(axis=1)

# drop RIPE RIS, change index type to int, name index to ASN, round data, calculate total bias score
new_bias_diff.drop('RIPE RIS',inplace=True)
new_bias_diff.index = new_bias_diff.index.astype('float')
new_bias_diff.index = new_bias_diff.index.astype('int')
new_bias_diff.index.name = 'ASN'
new_bias_diff = new_bias_diff.round(2)
# new_bias_diff['TOTAL bias'] = new_bias_diff.sum(axis=1)


'''
Join bias dataframe with AI4NetMon dataframe
'''
new_bias_diff = new_bias_diff.join(df)
new_bias_diff.to_csv(BIAS_DF_GROUPS.format('csv'))
new_bias_diff.to_json(BIAS_DF_GROUPS.format('json'))