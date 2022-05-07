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

ONLY_FULL_FEEDERS = True

if ONLY_FULL_FEEDERS:
	filestr = '_FullFeeders'
	
	ASN2ASN_DIST_FNAME = '../../data/misc/asn2asn__only_peers_pfx.json'
	RIS_IP2ASN = '../../data/misc/RIPE_RIS_peers_ip2asn.json'
	print('Loading proximity dict...')
	with open(ASN2ASN_DIST_FNAME, 'r') as f:
		asn2asn = json.load(f)
	feed = defaultdict(lambda : 0)
	for o_asn, dict_o_asn in asn2asn.items():
		for m_asn, dist in dict_o_asn.items():
			feed[m_asn] +=1
	full_feeders_ips = [m_asn for m_asn, nb_feeds in feed.items() if nb_feeds > 65000]
	
	with open(RIS_IP2ASN, 'r') as f:
		ris_dict = json.load(f)
	full_feeders = [ris_dict[ip] for ip in full_feeders_ips if ip in ris_dict.keys()]
else:
	filestr = ''


## datasets
FNAME_SORTED_LIST_NAIVE  = './data/sorted_list_naive.json'
FNAME_SORTED_LIST_GREEDY = './data/sorted_list_greedy.json'
FNAME_SORTED_BIAS_NAIVE  = './data/sorted_bias_naive{}.csv'.format(filestr)
FNAME_SORTED_BIAS_GREEDY = './data/sorted_bias_greedy{}.csv'.format(filestr)


# select features for visualization
FEATURE_NAMES_DICT = bu.get_features_dict_for_visualizations() 
FEATURES = list(FEATURE_NAMES_DICT.keys())


## load data
df = dat.load_aggregated_dataframe(preprocess=True)
df.index = df.index.astype(str,copy=False)
print(df)

df_ris = df.loc[(df['is_ris_peer_v4']>0) | (df['is_ris_peer_v6']>0)]
ris_asns = list(df_ris.index)
non_ris_asns = list(set(df.index)-set(ris_asns))

with open(FNAME_SORTED_LIST_NAIVE, 'r') as f:
	naive_sel = json.load(f)
with open(FNAME_SORTED_LIST_GREEDY, 'r') as f:
	greedy_sel = json.load(f)



params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}

if ONLY_FULL_FEEDERS:
	set_of_monitors = set([i for i in set(df_ris.index) if int(float(i)) in full_feeders])
	naive_sel = [i for i in naive_sel if i in set_of_monitors]
	greedy_sel = [i for i in greedy_sel if i in set_of_monitors]
else:
	set_of_monitors = set(df_ris.index)


naive_bias = []
greedy_bias = []
for i in tqdm(range(len(naive_sel))):
	dd = bu.bias_score_dataframe(df[FEATURES], {'naive': df.loc[set_of_monitors- set(naive_sel[0:i]),FEATURES], 
												'greedy':df.loc[set_of_monitors- set(greedy_sel[0:i]),FEATURES]}, **params).sum(axis=0)
	naive_bias.append(dd['naive'])
	greedy_bias.append(dd['greedy'])

with open(FNAME_SORTED_BIAS_NAIVE,'w') as f:
	json.dump(naive_bias,f)

with open(FNAME_SORTED_BIAS_GREEDY,'w') as f:
	json.dump(greedy_bias,f)


# for i in tqdm(range(len(naive_sel))):
# 	dd = bu.bias_score_dataframe(df[FEATURES], {'naive': df.loc[set_of_monitors- set(naive_sel[0:i]),FEATURES], 
# 												'greedy':df.loc[set_of_monitors- set(greedy_sel[0:i]),FEATURES]}, **params).sum(axis=0)
# 	if i>0:
# 		with open(FNAME_SORTED_BIAS_NAIVE,'r') as f: 
# 			d = json.load(f)
# 			d.append(dd['naive'])
# 	else:
# 		d = [dd['naive']]
# 	with open(FNAME_SORTED_BIAS_NAIVE,'w') as f:
# 		json.dump(d,f)
# 	if i>0:
# 		with open(FNAME_SORTED_BIAS_GREEDY,'r') as f: 
# 			d = json.load(f)
# 			d.append(dd['greedy'])
# 	else:
# 		d = [dd['greedy']]
# 	with open(FNAME_SORTED_BIAS_GREEDY,'w') as f:
# 		json.dump(d,f)