import numpy as np
import pandas as pd
import json
import random
from matplotlib import pyplot as plt
from ai4netmon.Analysis.bias import bias_utils as bu
from ai4netmon.Analysis.bias import radar_chart

## datasets
AGGREGATE_DATA_FNAME = '../../data/aggregate_data/asn_aggregate_data_20211201.csv'
BIAS_DF_SAVE_FNAME = './data/bias_df__no_stubs.csv'
BIAS_TOTAL_DIFF_SAVE_FNAME = './data/bias_total_df__no_stubs.csv'

# select features for visualization
FEATURE_NAMES_DICT = bu.get_features_dict_for_visualizations() 
FEATURES = list(FEATURE_NAMES_DICT.keys())


## load data
df = pd.read_csv(AGGREGATE_DATA_FNAME, header=0, index_col=0)
df['is_personal_AS'].fillna(0, inplace=True)
df = df[df['AS_rel_degree']>1]	# remove stub ASes

df_ris = df.loc[(df['is_ris_peer_v4']>0) | (df['is_ris_peer_v6']>0)]
ris_asns = list(df_ris.index)
non_ris_asns = list(set(df.index)-set(ris_asns))


## calculate bias for all features
network_sets_dict_for_bias = dict()
network_sets_dict_for_bias['RIPE RIS'] = df_ris[FEATURES]
for ASN in non_ris_asns:
	network_sets_dict_for_bias[str(ASN)] = df.loc[ris_asns + [ASN]][FEATURES]

params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}
bias_df = bu.bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **params)

bias_diff = bias_df.sum(axis=0)

bias_df.to_csv(BIAS_DF_SAVE_FNAME, index=True, header=True)
bias_diff.sort_values().to_csv(BIAS_TOTAL_DIFF_SAVE_FNAME, index=True, header=True)