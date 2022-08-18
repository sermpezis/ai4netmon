from ai4netmon.Analysis.aggregate_data import data_aggregation_tools as dat
from ai4netmon.Analysis.bias import bias_utils as bu
from tqdm import tqdm
import pandas as pd 

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

params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}

## Atlas
network_sets_dict_for_bias = dict()
network_sets_dict_for_bias['Atlas'] = df_atlas[FEATURES]
bias_df = pd.DataFrame()
i = 1
for ASN in tqdm(non_atlas_asns):
	network_sets_dict_for_bias[str(ASN)] = df.loc[atlas_asns + [ASN]][FEATURES]
	if not i%1000:
		bias_df = pd.concat([bias_df, bu.bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **params)], axis=1)
		network_sets_dict_for_bias = dict()
	i+=1

# bias_df = bu.bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **params)
bias_df.to_csv(BIAS_DF_SAVE_FNAME.format('Atlas'), index=True, header=True)
# print(bias_df)