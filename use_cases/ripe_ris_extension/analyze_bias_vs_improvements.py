from ai4netmon.Analysis.aggregate_data import data_aggregation_tools as dat
from ai4netmon.Analysis.bias import bias_utils as bu
import pandas
from tqdm import tqdm
import json
import os
import pandas as pd 
from matplotlib import pyplot as plt


## datasets
BIAS_DF_SAVE_FNAME = './data/bias_df__no_stubs_{}.csv'
BIAS_DF_SAVE_FNAME = './data/bias_df__MINUS_ONE.csv'
DATA_FILES_FORMAT = '../../../MISC/feed_stats/data/data_per_asn_{}__only_links_comms.json'
BIAS_LINKS_COMMS_SAVE_FNAME = './data/bias_df__MINUS_ONE__links_comms.csv'
FIG_BIAS_LINKS_COMMS = './figures/fig_scatter_bias_vs_extra_{}.png'

# select features for visualization
FEATURE_NAMES_DICT = bu.get_features_dict_for_visualizations() 
FEATURES = list(FEATURE_NAMES_DICT.keys())



if not os.path.exists(BIAS_LINKS_COMMS_SAVE_FNAME):
	## load data
	df = dat.load_aggregated_dataframe(preprocess=True)
	df = df[df['AS_rel_degree']>1]	# remove stub ASes
	df.index = df.index.astype(float)
	df.index = df.index.astype(int)


	df_ris = df.loc[(df['is_ris_peer_v4']>0) | (df['is_ris_peer_v6']>0)]
	ris_asns = list(df_ris.index)
	df_rv = df.loc[df['is_routeviews_peer']>0]
	rv_asns = list(df_rv.index)

	## load bias data
	bias_df = pd.read_csv(BIAS_DF_SAVE_FNAME, header=0, index_col=0)
	bias_df_sum = bias_df.sum(axis=0).sort_values()
	bias_df_diff_rel = (bias_df_sum / bias_df_sum['RIPE RIS'] - 1)*100
	bias_df_diff_rel = pd.DataFrame(bias_df_diff_rel)
	bias_df_diff_rel.rename(columns={0:'bias'}, inplace=True)
	bias_df_diff_rel.drop('RIPE RIS', inplace=True)
	bias_df_diff_rel.index = bias_df_diff_rel.index.astype(float)
	bias_df_diff_rel.index = bias_df_diff_rel.index.astype(int)



	dict_links = dict()
	dict_comms = dict()

	print('Loading all ASN data')
	for ASN in tqdm(ris_asns):
		datafile = DATA_FILES_FORMAT.format(ASN)
		if os.path.exists(datafile):
			with open(datafile, 'r') as f:
				d = json.load(f)
			dict_links[ASN] = d['dlinks4']
			dict_comms[ASN] = d['comms4']
			print('file: \t\t'+str(ASN))
		else:
			print('No file: '+str(ASN))

	for ASN in tqdm(dict_links.keys()):
		datafile = DATA_FILES_FORMAT.format(ASN)
		with open(datafile, 'r') as f:
			d = json.load(f)
		seen_links = []
		seen_comms = []
		for m in dict_links.keys():
			if m==ASN:
				continue
			seen_links.extend(dict_links[m])
			seen_comms.extend(dict_links[m])
		bias_df_diff_rel.loc[ASN,'links'] = len(set(d['dlinks4']) - set(seen_links))
		bias_df_diff_rel.loc[ASN,'comms'] = len(set(d['comms4']) - set(seen_comms))


	bias_df_diff_rel.to_csv(BIAS_LINKS_COMMS_SAVE_FNAME)
else:
	bias_df_diff_rel = pd.read_csv(BIAS_LINKS_COMMS_SAVE_FNAME)

	
plt.scatter(bias_df_diff_rel['bias'],bias_df_diff_rel['links'])
plt.xlabel('relative difference (%) in bias score', fontsize=FONTSIZE)
plt.ylabel('nb. extra AS links seen', fontsize=FONTSIZE)
plt.xticks(fontsize=FONTSIZE)
plt.yticks(fontsize=FONTSIZE)
plt.grid(True)
plt.subplots_adjust(left=0.15, bottom=0.15)
plt.savefig(FIG_BIAS_LINKS_COMMS.format('links'))
plt.close()


plt.scatter(bias_df_diff_rel['bias'],bias_df_diff_rel['comms'])
plt.xlabel('relative difference (%) in bias score', fontsize=FONTSIZE)
plt.ylabel('nb. extra BGP communities seen', fontsize=FONTSIZE)
plt.xticks(fontsize=FONTSIZE)
plt.yticks(fontsize=FONTSIZE)
plt.grid(True)
plt.subplots_adjust(left=0.15, bottom=0.15)
plt.savefig(FIG_BIAS_LINKS_COMMS.format('comms'))
plt.close()

