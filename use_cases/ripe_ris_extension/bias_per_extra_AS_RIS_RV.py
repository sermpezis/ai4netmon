from ai4netmon.Analysis.aggregate_data import data_aggregation_tools as dat
from ai4netmon.Analysis.bias import bias_utils as bu
from tqdm import tqdm

## datasets
BIAS_DF_SAVE_FNAME = './data/bias_df__no_stubs_{}.csv'

# select features for visualization
FEATURE_NAMES_DICT = bu.get_features_dict_for_visualizations() 
FEATURES = list(FEATURE_NAMES_DICT.keys())


## load data
df = dat.load_aggregated_dataframe(preprocess=True)
df = df[df['AS_rel_degree']>1]	# remove stub ASes

df_ris = df.loc[(df['is_ris_peer_v4']>0) | (df['is_ris_peer_v6']>0)]
ris_asns = list(df_ris.index)
non_ris_asns = list(set(df.index)-set(ris_asns))
df_rv = df.loc[df['is_routeviews_peer']>0]
rv_asns = list(df_rv.index)
non_rv_asns = list(set(df.index)-set(rv_asns))

df_atlas = df.loc[(df['nb_atlas_probes_v4']>0) | (df['nb_atlas_probes_v6']>0)]
atlas_asns = list(df_atlas.index)
non_atlas_asns = list(set(df.index)-set(atlas_asns))


## Atlas
network_sets_dict_for_bias = dict()
network_sets_dict_for_bias['Atlas'] = df_atlas[FEATURES]
for ASN in non_atlas_asns:
	network_sets_dict_for_bias[str(ASN)] = df.loc[atlas_asns + [ASN]][FEATURES]

params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}
bias_df = bu.bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **params)
bias_df.to_csv(BIAS_DF_SAVE_FNAME.format('Atlas'), index=True, header=True)


## RIS
network_sets_dict_for_bias = dict()
network_sets_dict_for_bias['RIS'] = df_ris[FEATURES]
for ASN in non_ris_asns:
	network_sets_dict_for_bias[str(ASN)] = df.loc[ris_asns + [ASN]][FEATURES]

params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}
bias_df = bu.bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **params)
bias_df.to_csv(BIAS_DF_SAVE_FNAME.format('RIS'), index=True, header=True)


# ## RV
# network_sets_dict_for_bias = dict()
# network_sets_dict_for_bias['RV'] = df_rv[FEATURES]
# for ASN in tqdm(non_rv_asns):
# 	network_sets_dict_for_bias[str(ASN)] = df.loc[rv_asns + [ASN]][FEATURES]
# params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}
# bias_df = bu.bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **params)
# bias_df.to_csv(BIAS_DF_SAVE_FNAME.format('RV'), index=True, header=True)