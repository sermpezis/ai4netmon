import pandas as pd
from matplotlib import pyplot as plt
from ai4netmon.Analysis.bias import bias_utils as bu
from ai4netmon.Analysis.bias import radar_chart
from ai4netmon.Analysis.aggregate_data import data_aggregation_tools as dat
import time
from tqdm import tqdm
import csv
import json 
from collections import defaultdict
import numpy as np

## datasets
FNAME_SORTED_LIST_NAIVE  = './data/sorted_list_naive_Atlas.json'
FNAME_SORTED_BIAS_NAIVE  = './data/sorted_bias_naive_Atlas.csv'
FNAME_SORTED_LIST_GREEDY  = './data/sorted_list_greedy_Atlas.json'
FNAME_SORTED_BIAS_GREEDY  = './data/sorted_bias_greedy_Atlas.csv'


# select features for visualization
FEATURE_NAMES_DICT = bu.get_features_dict_for_visualizations() 
FEATURES = list(FEATURE_NAMES_DICT.keys())


## load data
df = dat.load_aggregated_dataframe(preprocess=True)
df.index = df.index.astype(str,copy=False)
print(df)

df_atlas = df.loc[(df['nb_atlas_probes_v4']>0) | (df['nb_atlas_probes_v6']>0)]


with open(FNAME_SORTED_LIST_NAIVE, 'r') as f:
	naive_sel = json.load(f)
with open(FNAME_SORTED_LIST_GREEDY, 'r') as f:
	greedy_sel = json.load(f)



params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}


set_of_monitors = set(df_atlas.index)


naive_bias = []
greedy_bias = []
for i in tqdm(range(len(naive_sel))):
	# dd = bu.bias_score_dataframe(df[FEATURES], {'naive': df.loc[set_of_monitors- set(naive_sel[0:i]),FEATURES], 
	# 											'greedy':df.loc[set_of_monitors- set(greedy_sel[0:i]),FEATURES]}, **params).sum(axis=0)
	# dd = bu.bias_score_dataframe(df[FEATURES], {'naive': df.loc[set_of_monitors- set(naive_sel[0:i]),FEATURES]}, **params).sum(axis=0)
	# naive_bias.append(dd['naive'])
	if i <= len(greedy_sel):
		dd = bu.bias_score_dataframe(df[FEATURES], {'greedy':df.loc[set_of_monitors- set(greedy_sel[0:i]),FEATURES]}, **params).sum(axis=0)
		greedy_bias.append(dd['greedy'])
	else:
		greedy_bias.append(np.nan)

# with open(FNAME_SORTED_BIAS_NAIVE,'w') as f:
# 	json.dump(naive_bias,f)

with open(FNAME_SORTED_BIAS_GREEDY,'w') as f:
	json.dump(greedy_bias,f)
