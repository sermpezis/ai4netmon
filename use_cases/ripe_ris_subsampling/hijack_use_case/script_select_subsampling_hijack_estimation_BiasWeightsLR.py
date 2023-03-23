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
LIST_OF_MONITORS = './data/list_of_monitors.json'
FNAME_SORTED_LIST_NAIVE  = './data/sorted_list_naive_BiasWeightsLR_hijack_{}.json'
FNAME_SORTED_LIST_GREEDY = './data/sorted_list_greedy_BiasWeightsLR_hijack_{}.json'
FNAME_SORTED_BIAS_NAIVE  = './data/sorted_bias_naive_BiasWeightsLR_hijack_{}.csv'
FNAME_SORTED_BIAS_GREEDY = './data/sorted_bias_greedy_BiasWeightsLR_hijack_{}.csv'

# select features for visualization
FEATURE_NAMES_DICT = bu.get_features_dict_for_visualizations() 
FEATURES = list(FEATURE_NAMES_DICT.keys())


## load list of monitors
with open(LIST_OF_MONITORS, 'r') as f:
	list_of_monitors = json.load(f)

## load data
df = dat.load_aggregated_dataframe(preprocess=True)
df.index = df.index.astype(int,copy=False)
df.index = df.index.astype(str,copy=False)
# print(df)


set_infra = dict()
params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}


### @Sofia: set in the following code the bias weights from the LR in the format of the example dictionary
bias_weights = {'AS_rank_iso': 0.5, 'dimension 2': 0.9, ...}
### @Sofia: end of added code

for k in ['RC', 'RA']:
	print(k)
	set_infra[k] = df.loc[df.index.isin( list_of_monitors[k] ),:]	
	set_of_monitors = set(set_infra[k].index)

	print('\t Sorting algorithm...')
	a = bu.subsampling('sorting', df, set_of_monitors, params, FEATURES, bias_weights)
	with open(FNAME_SORTED_LIST_NAIVE.format(k),'w') as f:
		json.dump(a,f)

	print('\t Greedy algorithm ...')
	b = bu.subsampling('greedy', df, set_of_monitors, params, FEATURES, bias_weights)
	with open(FNAME_SORTED_LIST_GREEDY.format(k),'w') as f:
		json.dump(b,f)

	# for i in range(len(a)):
	# 	dd = bu.bias_score_dataframe(df[FEATURES], {'sorting':df.loc[set_of_monitors- set(a[0:i]),FEATURES], 'greedy':df.loc[set_of_monitors- set(b[0:i]),FEATURES]}, **params).sum(axis=0)
	# 	if i>0:
	# 		with open(FNAME_SORTED_BIAS_NAIVE.format(k),'r') as f: 
	# 			d = json.load(f)
	# 			d.append(dd['sorting'])
	# 	else:
	# 		d = [dd['sorting']]
	# 	with open(FNAME_SORTED_BIAS_NAIVE.format(k),'w') as f:
	# 		json.dump(d,f)
	# 	if i>0:
	# 		with open(FNAME_SORTED_BIAS_GREEDY.format(k),'r') as f: 
	# 			d = json.load(f)
	# 			d.append(dd['greedy'])
	# 	else:
	# 		d = [dd['greedy']]
	# 	with open(FNAME_SORTED_BIAS_GREEDY.format(k),'w') as f:
	# 		json.dump(d,f)