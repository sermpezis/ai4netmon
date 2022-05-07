import pandas as pd
from matplotlib import pyplot as plt
from ai4netmon.Analysis.bias import bias_utils as bu
from ai4netmon.Analysis.bias import radar_chart
from ai4netmon.Analysis.aggregate_data import data_aggregation_tools as dat
import time
from tqdm import tqdm
import csv
import json 
import random
## datasets

# select features for visualization
FEATURE_NAMES_DICT = bu.get_features_dict_for_visualizations() 
FEATURES = list(FEATURE_NAMES_DICT.keys())


## load data
df = dat.load_aggregated_dataframe(preprocess=True)
df.index = df.index.astype(str,copy=False)
print(df)

sets = dict()
sets['Atlas ALL'] = list(df.loc[(df['nb_atlas_probes_v4']>0) | (df['nb_atlas_probes_v6']>0)].index)


with open('./data/sorted_list_naive_Atlas.json', 'r') as f:
	naive_sel = json.load(f)
sets['Atlas sub 1000'] = naive_sel[-1000:]


with open('./data/sorted_list_greedy_rnd300_Atlas.json', 'r') as f:
	greedy_sel = json.load(f)
# sets['Atlas rnd 300'] = greedy_sel
sets['Atlas sub 100'] = greedy_sel[-100:]
sets['Atlas rnd 100'] = random.sample(sets['Atlas ALL'],100)



params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}




bias_df = bu.bias_score_dataframe(df[FEATURES], {k:df.loc[v,FEATURES] for k,v in sets.items()}, **params).mean(axis=0)

print(bias_df)


sets_int = dict()
for k,v in sets.items():
	sets_int[k] = [int(float(i)) for i in v]
with open('Lists_of_Atlas_samples.json', 'w') as f:
	json.dump(sets_int,f)
