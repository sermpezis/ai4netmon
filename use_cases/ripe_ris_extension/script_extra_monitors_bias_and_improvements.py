import pandas as pd
from ai4netmon.Analysis.bias import generate_distribution_plots as gdp
from matplotlib import pyplot as plt
from matplotlib import cm, colors


AGGREGATE_DATA_FNAME = 'https://raw.githubusercontent.com/sermpezis/ai4netmon/dev/data/aggregate_data/asn_aggregate_data_20211201.csv'
BIAS_TOTAL_DF = './data/bias_total_df__no_stubs.csv'
IMPROVEMENTS = '../../Datasets/improvements20210601.txt'
ORDERED_LIST_BIASES = './data/sorted_asns_by_ascending_biases__no_stubs.json'
SAVE_PLOTS_FNAME_FORMAT = './figures/Fig_extra_monitors_{}_{}'
SAVE_CSV_FNAME = './data/df_bias_vs_improvement.csv'
PLOT_SCATTER_FNAME = './figures/Fig_scatter_bias_vs_improvement.png'


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

# custom function for 2d visualization
FONTSIZE = 15
FONTSIZE_SMALL = 10

def plot_2d_visualization(df, viz_items, colors_property, line=None, save_csv=True):
    if len(viz_items)==0:
        df_plot = df
    else:
        df_plot = df[df[colors_property].isin(viz_items)]

    unique_values = pd.unique(df_plot[colors_property]).tolist()
    # print(unique_values)
    dict_colors = {v:i for i,v in enumerate(unique_values)}
    df_plot['Colors'] = [dict_colors.get(v) for v in df_plot[colors_property]]
    if save_csv:
        df_plot['ASN'] = df_plot.index
        df_plot.to_csv(SAVE_CSV_FNAME, index=False)


    cmap = cm.get_cmap('Set1')
    norm = colors.Normalize(vmin=0, vmax=len(unique_values))
    norm_color = lambda x: cmap(norm(x))

    fig, ax = plt.subplots()
    for k,v in dict_colors.items():
        d = df_plot[df_plot[colors_property]==k]
        ax.scatter(d['X'], d['Y'], s=10, color=norm_color(v), label=k, cmap='Set1')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(loc='upper left', fontsize=FONTSIZE_SMALL)
    plt.xlabel('Total bias difference (%)', fontsize=FONTSIZE)
    plt.ylabel('Improvement', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.grid(True)
    plt.yscale('log')
    if line is not None:
        plt.vlines(line, df_plot['Y'].min(), df_plot['Y'].max(), colors='k',linewidth=3)
    plt.savefig(PLOT_SCATTER_FNAME)
    plt.show()
    plt.close()



# plot scatter bias vs. improvement
continents = ['Asia', 'Europe', 'Africa', 'North America', 'Oceania', 'South America']
continents = []
plot_2d_visualization(df_plot.rename({'bias':'X', 'improvement':'Y'},axis=1), continents, 'AS_rank_continent', line=0)



## load ordered list of extra monitors and plot their characteristics
order_bias = pd.read_json(ORDERED_LIST_BIASES, orient='records')['total'].to_list()

dict_network_sets = dict()
dict_network_sets['all'] = df
dict_network_sets['RIPE RIS (all)'] = df.loc[(df['is_ris_peer_v4']>0) | (df['is_ris_peer_v6']>0)]
dict_network_sets['+50'] = df.loc[order_bias[0:50]]
dict_network_sets['+200'] = df.loc[order_bias[0:200]]

gdp.plot_all(dict_network_sets, SAVE_PLOTS_FNAME_FORMAT, save_json=False, show_plot=False)








