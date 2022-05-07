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
df.index = df.index.astype(float,copy=False)
df.index = df.index.astype(int,copy=False)
print(df)

sets = dict()
sets['Atlas ALL'] = list(df.loc[(df['nb_atlas_probes_v4']>0) | (df['nb_atlas_probes_v6']>0)].index)

with open('./data/sorted_list_greedy_Atlas.json', 'r') as f:
	greedy_sel = json.load(f)
sets = dict()
sets['Atlas greedy 1000'] = [int(float(i)) for i in greedy_sel[-1000:]]
sets['Atlas greedy 300'] = [int(float(i)) for i in greedy_sel[-300:]]
sets['Atlas rnd 300'] = [int(float(j)) for j in random.sample(greedy_sel,300)]

params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}

bias_df = bu.bias_score_dataframe(df[FEATURES], {k:df.loc[v,FEATURES] for k,v in sets.items()}, **params).mean(axis=0)

print(bias_df)

with open('Lists_of_Atlas_samples_Greedy.json', 'w') as f:
	json.dump(sets,f)
