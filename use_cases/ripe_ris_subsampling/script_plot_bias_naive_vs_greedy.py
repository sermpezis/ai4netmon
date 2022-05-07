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


FNAME_SORTED_BIAS_NAIVE  = './data/sorted_bias_naive_{}.csv'
FNAME_SORTED_BIAS_GREEDY = './data/sorted_bias_greedy_{}.csv'
SAVEFIG = './fig_bias_vs_sampling_naive_and_greedy_{}.png'


# select features for visualization
FEATURE_NAMES_DICT = bu.get_features_dict_for_visualizations() 
FEATURES = list(FEATURE_NAMES_DICT.keys())


set_infra = ['RIS','FullFeeders','RV','Atlas', 'rnd300_Atlas']
colors_infra = {'RIS':'b', 'FullFeeders':'k', 'RV':'r', 'Atlas':'g', 'rnd300_Atlas':'m'}
labels_infra = {'RIS':'RIPE RIS', 'FullFeeders':'RIPE RIS (full feeders)', 'RV':'RouteViews', 'Atlas':'RIPE Atlas', 'rnd300_Atlas':'Atlas RND'}

for infra in ['Atlas', 'rnd300_Atlas']:#['RIS','RV', 'Atlas', 'rnd300_Atlas']:
	with open(FNAME_SORTED_BIAS_NAIVE.format(infra), 'r') as f:
		naive_bias = json.load(f)
	try:
		with open(FNAME_SORTED_BIAS_GREEDY.format(infra), 'r') as f:
			greedy_bias = json.load(f)
	except:
		greedy_bias = [np.nan]*len(naive_bias)

	naive_bias = [i/len(FEATURES) for i in naive_bias]
	greedy_bias = [i/len(FEATURES) for i in greedy_bias]
	x_vector = [len(naive_bias)-1-i for i in range(len(naive_bias))]

	print(infra)
	print('[Greedy] Min value is {} at #monitors: {}'.format(np.min(greedy_bias), x_vector[np.argmin(greedy_bias)]))
	print('[Naive] Min value is {} at #monitors: {}'.format(np.min(naive_bias), x_vector[np.argmin(naive_bias)]))

	naive_linestyle = '--{}'.format(colors_infra[infra])
	greedy_linestyle = '{}'.format(colors_infra[infra])
	naive_label = '{} (naive)'.format(labels_infra[infra])
	greedy_label = '{} (greedy)'.format(labels_infra[infra])
	plt.plot(x_vector, naive_bias, naive_linestyle, label=naive_label)
	plt.plot(x_vector, greedy_bias, greedy_linestyle, label=greedy_label)





FONTSIZE = 15
plt.axis([9,4000,0,0.40])
plt.xscale('log')
plt.xlabel('#monitors', fontsize=FONTSIZE)
plt.ylabel('Bias score', fontsize=FONTSIZE)
plt.xticks(fontsize=FONTSIZE)
plt.yticks(fontsize=FONTSIZE)
plt.legend(fontsize=FONTSIZE, loc='upper right')
plt.grid(True)
plt.subplots_adjust(left=0.15, bottom=0.15)
# plt.yticks([0.05*i for i in range(11)])
# plt.xticks([10*i for i in range(10)]+[100+i*100 for i in range])
# plt.axis([-10,len(naive_bias)+10,0,0.5])
plt.savefig(SAVEFIG.format('only_Atlas'))
plt.show()
plt.close()