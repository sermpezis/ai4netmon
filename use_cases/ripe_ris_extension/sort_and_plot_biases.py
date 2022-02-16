import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import json

BIAS_DF = './data/bias_df__no_stubs.csv'
BIAS_DIFF = './data/bias_total_df__no_stubs.csv'
SAVEFIG_FORMAT = './figures/cdf_bias_diff_{}.png'
SAVE_JSON_FNAME = './data/sorted_asns_by_ascending_biases.json'

bias_df = pd.read_csv(BIAS_DF, header=0, index_col=0)
print(bias_df)

bias_diff = pd.read_csv(BIAS_DIFF, index_col=0)
print(bias_diff)


## plotting parameters
FONTSIZE = 15
LINEWIDTH = 2
MARKERSIZE = 10


def plot_cdf_bias(plot_df, feature_name, show_plot=False):
	bias_diff_ris = plot_df.loc['RIPE RIS']
	cdf = ECDF(plot_df)
	plt.plot(cdf.x, cdf.y)
	plt.plot([bias_diff_ris, bias_diff_ris], [0,1], 'r', linewidth=LINEWIDTH)
	plt.axis([min(plot_df), max(plot_df), 0, 1])
	plt.legend(['CDF (RIPE RIS + 1 ASN)', 'RIPE RIS'], fontsize=FONTSIZE)
	plt.xlabel('{} bias (RIPE RIS + ASN)'.format(feature_name), fontsize=FONTSIZE)
	plt.ylabel('CDF', fontsize=FONTSIZE)
	plt.xticks(fontsize=FONTSIZE)
	plt.yticks(fontsize=FONTSIZE)
	plt.grid(True)
	plt.savefig(SAVEFIG_FORMAT.format(feature_name))
	if show_plot:
		plt.show()
	plt.close()

def get_sorted_list(plot_df):
	df = plot_df.sort_values()
	return [int(float(i)) for i in df.index if i != 'RIPE RIS']




# plot total bias
plot_df = bias_diff['0']
plot_cdf_bias(plot_df, 'total')


ris_bias = bias_diff.loc['RIPE RIS','0']
plot_df = (bias_diff['0'] - ris_bias) / ris_bias * 100
plot_cdf_bias(plot_df, 'total difference %')



dict_of_biases = dict()
dict_of_biases['total'] = get_sorted_list(plot_df)
# plot partial bias
for feature in bias_df.index:
	plot_df = bias_df.loc[feature]
	plot_cdf_bias(plot_df, feature)
	dict_of_biases[feature] = get_sorted_list(plot_df)

with open(SAVE_JSON_FNAME, 'w') as f:
	json.dump(dict_of_biases, f)