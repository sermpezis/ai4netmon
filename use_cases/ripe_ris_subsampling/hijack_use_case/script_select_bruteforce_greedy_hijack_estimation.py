import pandas as pd
import numpy as np
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
HIJACK_SIMS = './data/impact__CAIDA20190801_sims2000_hijackType0_per_monitor_onlyRC_NEW_with_mon_ASNs.csv'
FNAME_SORTED_LIST  = './data/sorted_list_bruteforce_greedy_hijack_{}.json'
FNAME_SORTED_BIAS  = './data/sorted_bias_bruteforce_greedy_PartialBias_hijack_{}.csv'





df = pd.read_csv(HIJACK_SIMS)
df = df.fillna(0.5)
df['actual_impact'] = df['-1.4'] / df['-1.2']
df['estimated_impact'] = df['-1.6'] / df['-1.5']
df = df[df['actual_impact']<=1]
df = df.iloc[:,7:]

def calculate_set_error(monitors):
	return np.sqrt(((df[monitors].mean(axis=1) - df['actual_impact']) ** 2).mean())

def get_argmax_error(monitors):
	rmse_wo_mon = pd.DataFrame(index=monitors)
	for mon in monitors:
		rmse_wo_mon.loc[mon,'error'] = calculate_set_error(set(monitors)-set([mon]))
	return rmse_wo_mon['error'].idxmin(), rmse_wo_mon['error'].min()

def bruteforce_greedy(monitors):
	current_set = set(monitors)
	ordered_list = []
	while len(current_set)>1:
		ASN, err = get_argmax_error(current_set)
		print(len(current_set), round(err,3))
		ordered_list.append(ASN)
		current_set.remove(ASN)
	ordered_list.append(list(current_set)[0])
	return ordered_list

ordered_list = bruteforce_greedy(df.columns[0:-2])

k = 'RC'
with open(FNAME_SORTED_LIST.format(k),'w') as f:
	json.dump(ordered_list,f)





# select features for visualization
FEATURE_NAMES_DICT = bu.get_features_dict_for_visualizations() 
FEATURES = list(FEATURE_NAMES_DICT.keys())

## load data
df = dat.load_aggregated_dataframe(preprocess=True)
df.index = df.index.astype(int,copy=False)
df.index = df.index.astype(str,copy=False)

params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}

for i in range(len(ordered_list)):
	dd = bu.bias_score_dataframe(df[FEATURES], {'sorting':df.loc[ordered_list[i+1:],FEATURES]}, **params).sum(axis=0)
	if i>0:
		with open(FNAME_SORTED_BIAS.format(k),'r') as f: 
			d = json.load(f)
			d.append(dd['sorting'])
	else:
		d = [dd['sorting']]
	with open(FNAME_SORTED_BIAS.format(k),'w') as f:
		json.dump(d,f)