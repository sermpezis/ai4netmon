import pandas as pd
from ai4netmon.Analysis.bias import generate_distribution_plots as gdp
from matplotlib import pyplot as plt
from matplotlib import cm, colors
from ai4netmon.Analysis.bias import bias_utils as bu
from ai4netmon.Analysis.bias import radar_chart


AGGREGATE_DATA_FNAME = 'https://raw.githubusercontent.com/sermpezis/ai4netmon/dev/data/aggregate_data/asn_aggregate_data_20211201.csv'
BIAS_TOTAL_DF = './data/bias_total_df__no_stubs.csv'
IMPROVEMENTS = '../../data/misc/improvements20210601.txt'
ORDERED_LIST_BIASES = './data/sorted_asns_by_ascending_biases.json'

IMPROVEMENT_THRESHOLD = 1000
SAVE_PLOTS_FNAME_FORMAT = './figures/Fig_extra_monitors_filtered'+str(IMPROVEMENT_THRESHOLD)+'_{}_{}'
FIG_RADAR_FNAME = './figures/fig_radar__ordered_filtered{}.png'.format(IMPROVEMENT_THRESHOLD)



## load network data
df = pd.read_csv(AGGREGATE_DATA_FNAME, header=0, index_col=0)
df['is_personal_AS'].fillna(0, inplace=True)

## load improvement data
df_imp = pd.read_csv(IMPROVEMENTS, sep=" ", names=['loc', 'IPv', 'ASN', 'improvement'])
df_imp.set_index('ASN', inplace=True)
df_imp = df_imp[(df_imp['loc']=='GLOBAL') & (df_imp['IPv']==4)]
df_imp = df_imp.loc[~df_imp.index.str.contains('{'),:]
df_imp.index = [int(i) for i in df_imp.index]
df_imp.index = df_imp.index.astype(float)
df_imp.index = df_imp.index.astype(int)

## load bias data 
df_bias = pd.read_csv(BIAS_TOTAL_DF, skiprows=1, names=['ASN', 'bias'])
df_bias.set_index('ASN', inplace=True)
ripe_bias = float(df_bias.loc['RIPE RIS'])
df_bias.drop(index=['RIPE RIS'], inplace=True)
df_bias.index = df_bias.index.astype(float)
df_bias.index = df_bias.index.astype(int)
df_bias = (df_bias - ripe_bias)/ripe_bias * 100


# create dataframe from existing info for plotting
df_plot = pd.merge(df_imp,df_bias, left_index=True, right_index=True)
df_plot = pd.merge(df_plot,df['AS_rank_continent'], how='left', left_index=True, right_index=True)

## load ordered list of extra monitors and plot their characteristics
order_bias = pd.read_json(ORDERED_LIST_BIASES, orient='records')['total'].to_list()
filtered_order_bias = [i for i in order_bias if (i in df_plot.index) and (df_plot.loc[i,'improvement']>IMPROVEMENT_THRESHOLD)]

dict_network_sets = dict()
dict_network_sets['all'] = df
dict_network_sets['RIPE RIS'] = df.loc[(df['is_ris_peer_v4']>0) | (df['is_ris_peer_v6']>0)]
dict_network_sets['+50'] = df.loc[order_bias[0:50]]
dict_network_sets['+200'] = df.loc[order_bias[0:200]]
dict_network_sets['+50 (filtered)'] = df.loc[filtered_order_bias[0:50]]
dict_network_sets['+200 (filtered)'] = df.loc[filtered_order_bias[0:200]]
gdp.plot_all(dict_network_sets, SAVE_PLOTS_FNAME_FORMAT, save_json=False, show_plot=False)


# radar plot
FEATURE_NAMES_DICT = bu.get_features_dict_for_visualizations() 
FEATURES = list(FEATURE_NAMES_DICT.keys())
params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}
network_sets_dict = dict()
ripe_ris_list = list(df.loc[(df['is_ris_peer_v4']>0) | (df['is_ris_peer_v6']>0)].index)
network_sets_dict['RIPE RIS'] = df.loc[ripe_ris_list]
network_sets_dict['RIPE RIS +50'] = df.loc[ripe_ris_list+order_bias[0:50]]
network_sets_dict['RIPE RIS +50 (filtered)'] = df.loc[ripe_ris_list+filtered_order_bias[0:50]]

plot_bias_df = bu.bias_score_dataframe(df[FEATURES], network_sets_dict, **params)
radar_chart.plot_radar_from_dataframe(plot_bias_df, colors=None, frame='polygon', save_filename=FIG_RADAR_FNAME, varlabels=FEATURE_NAMES_DICT, show=False)