import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from ai4netmon.Analysis.aggregate_data import data_aggregation_tools as dat
from ai4netmon.Analysis.bias import bias_utils as bu


BIAS_DF = './data/bias_df__no_stubs_{}.csv'
FIG_CDF_SAVENAME = './figures/fig_cdf_bias_diff_rel__no_stubs.png'
FIG_BIAS_SAVENAME = './figures/fig_cdf_bias_vs_nb_extra_monitors.png'
MAX_EXTRA = 400

FONTSIZE = 15
LINEWIDTH = 2
colors_infra = {'all':'k', 'RIS':'b', 'RV':'r', 'Atlas':'g', 'RIS & RV':'m'}


FEATURE_NAMES_DICT = bu.get_features_dict_for_visualizations() 
FEATURES = list(FEATURE_NAMES_DICT.keys())

## load data
df = dat.load_aggregated_dataframe(preprocess=True)
df = df[df['AS_rel_degree']>1]	# remove stub ASes

df_rc = dict()
df_rc['RIS'] = df.loc[(df['is_ris_peer_v4']>0) | (df['is_ris_peer_v6']>0)]
df_rc['RV'] = df.loc[df['is_routeviews_peer']>0]

for rc in ['RIS', 'RV']:
	bias_df = pd.read_csv(BIAS_DF.format(rc), header=0, index_col=0)
	bias_df_sum = bias_df.sum(axis=0).sort_values()
	bias_df_sum.rename(index={'RIPE RIS':'RIS'},inplace=True)
	bias_df_diff_rel = (bias_df_sum / bias_df_sum[rc] - 1)*100

	## plot CDF of rel. difference in bias
	cdf = ECDF(bias_df_diff_rel)
	plt.plot(cdf.x, cdf.y, linewidth=LINEWIDTH, label=rc, color=colors_infra[rc])
# plt.axis([min(bias_df_diff_rel), max(bias_df_diff_rel), 0, 1])
plt.legend(fontsize=FONTSIZE)
plt.xlabel('relative difference (%) in bias score', fontsize=FONTSIZE)
plt.ylabel('CDF', fontsize=FONTSIZE)
plt.xticks(fontsize=FONTSIZE)
plt.yticks(fontsize=FONTSIZE)
plt.grid(True)
plt.savefig(FIG_CDF_SAVENAME)
plt.close()


for rc in ['RIS', 'RV']:
	bias_df = pd.read_csv(BIAS_DF.format(rc), header=0, index_col=0)
	bias_df_sum = bias_df.sum(axis=0).sort_values()
	bias_df_sum.rename(index={'RIPE RIS':'RIS'},inplace=True)
	bias_df_diff_rel = (bias_df_sum / bias_df_sum[rc] - 1)*100

	network_sets_dict_for_bias = dict()
	network_sets_dict_for_bias[rc] = df_rc[rc][FEATURES]
	list_of_monitors = list(df_rc[rc].index)
	for i, ASN in enumerate(list(bias_df_sum.index)):
		if i == MAX_EXTRA:
			break
		list_of_monitors.append(int(float(ASN)))
		network_sets_dict_for_bias['{} +{}'.format(rc, i+1)] = df.loc[list_of_monitors][FEATURES]
	params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}
	bias_extra_df = bu.bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **params)

	plt.plot(list(range(MAX_EXTRA+1)), bias_extra_df.mean(axis=0), linewidth=LINEWIDTH, label=rc, color=colors_infra[rc])


plt.legend(fontsize=FONTSIZE)
plt.axis([-10, MAX_EXTRA+10, 0, 0.2])
plt.xlabel('nb. extra monitors', fontsize=FONTSIZE)
plt.ylabel('avg. bias score', fontsize=FONTSIZE)
plt.xticks(fontsize=FONTSIZE)
plt.yticks(fontsize=FONTSIZE)
plt.grid(True)
plt.subplots_adjust(left=0.15, bottom=0.15)
plt.savefig(FIG_BIAS_SAVENAME)
plt.close()