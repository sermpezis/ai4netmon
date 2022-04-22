import pandas as pd
from matplotlib import pyplot as plt
from ai4netmon.Analysis.bias import bias_utils as bu
from ai4netmon.Analysis.bias import radar_chart
from ai4netmon.Analysis.aggregate_data import data_aggregation_tools as dat
import time
from tqdm import tqdm
import csv
import json 

## datasets
FNAME_SORTED_LIST_NAIVE  = './data/sorted_list_naive.json'
FNAME_SORTED_LIST_GREEDY = './data/sorted_list_greedy.json'
FNAME_SORTED_BIAS_NAIVE  = './data/sorted_bias_naive.csv'
FNAME_SORTED_BIAS_GREEDY = './data/sorted_bias_greedy.csv'

# select features for visualization
FEATURE_NAMES_DICT = bu.get_features_dict_for_visualizations() 
FEATURES = list(FEATURE_NAMES_DICT.keys())


## load data
t0 = time.time()
df = dat.load_aggregated_dataframe(preprocess=True)
df.index = df.index.astype(str,copy=False)
print(df)
t1 = time.time()

df_ris = df.loc[(df['is_ris_peer_v4']>0) | (df['is_ris_peer_v6']>0)]
ris_asns = list(df_ris.index)
non_ris_asns = list(set(df.index)-set(ris_asns))



params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}

def select_naive_sorting(df, set_of_monitors, bias_params):
	network_sets_dict_for_bias = dict()
	for ASN in set_of_monitors:
		network_sets_dict_for_bias[str(ASN)] = df.loc[list(set(set_of_monitors)-set([ASN]))][FEATURES]
	bias_df = bu.bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **bias_params)
	bias_diff = bias_df.sum(axis=0)
	return bias_diff.sort_values().index.tolist()

def select_greedy(df, set_of_monitors, bias_params, write_to_filename=None):
	current_set = set(set_of_monitors)
	ordered_list = []
	while len(current_set)>0:
		ASN = select_naive_sorting(df, current_set, bias_params)[0]
		ordered_list.append(ASN)
		current_set.remove(ASN)
		if write_to_filename is not None:
			if len(ordered_list)>1:
				with open(FNAME_SORTED_LIST_GREEDY,'r') as f: 
					d = json.load(f)
					d.append(ASN)
			else:
				d = [ASN]
			with open(FNAME_SORTED_LIST_GREEDY,'w') as f:
				json.dump(d,f)
		print(ASN)
	return ordered_list

set_of_monitors = set(df_ris.index)
a = select_naive_sorting(df, set_of_monitors, params)
with open(FNAME_SORTED_LIST_NAIVE,'w') as f:
	json.dump(a,f)

b = select_greedy(df, set_of_monitors, params, FNAME_SORTED_LIST_GREEDY)

for i in range(10):
	dd = bu.bias_score_dataframe(df[FEATURES], {'naive':df.loc[set_of_monitors- set(a[0:i]),FEATURES], 'greedy':df.loc[set_of_monitors- set(b[0:i]),FEATURES]}, **params).sum(axis=0)
	if i>0:
		with open(FNAME_SORTED_BIAS_NAIVE,'r') as f: 
			d = json.load(f)
			d.append(dd['naive'])
	else:
		d = [dd['naive']]
	with open(FNAME_SORTED_BIAS_NAIVE,'w') as f:
		json.dump(d,f)
	if i>0:
		with open(FNAME_SORTED_BIAS_GREEDY,'r') as f: 
			d = json.load(f)
			d.append(dd['greedy'])
	else:
		d = [dd['greedy']]
	with open(FNAME_SORTED_BIAS_GREEDY,'w') as f:
		json.dump(d,f)