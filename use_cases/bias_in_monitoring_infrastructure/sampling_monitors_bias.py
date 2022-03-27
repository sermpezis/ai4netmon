import numpy as np
import pandas as pd
import json
import random
from matplotlib import pyplot as plt
from ai4netmon.Analysis.bias import bias_utils as bu
# from ai4netmon.Analysis.bias import radar_chart
# from ai4netmon.Analysis.bias import generate_distribution_plots as gdp
import os


## datasets
AGGREGATE_DATA_FNAME = '../../data/aggregate_data/asn_aggregate_data_20211201.csv'
BIAS_CSV_FNAME = './data/bias_values_random_samples.csv'
BIAS_CSV_FNAME_NO_STUBS = './data/bias_values_random__no_stubs.csv'
OMIT_STUBS = False
if OMIT_STUBS:
    BIAS_CSV_FNAME = BIAS_CSV_FNAME_NO_STUBS
NB_SAMPLES = [10, 20, 50, 100, 200, 500, 1000]
NB_ITERATIONS = 100

mons = ['all', 'RIPE RIS (all)','RouteViews (all)','RIS + RV (all)','RIPE Atlas (all)']
if os.path.exists(BIAS_CSV_FNAME):
    bias_df = pd.read_csv(BIAS_CSV_FNAME, header=0, index_col=0)
else:
    # select features for visualization
    FEATURE_NAMES_DICT = bu.get_features_dict_for_visualizations() 
    FEATURES = list(FEATURE_NAMES_DICT.keys())


    ## load data
    df = pd.read_csv(AGGREGATE_DATA_FNAME, header=0, index_col=0)
    df['is_personal_AS'].fillna(0, inplace=True)
    # df = df[df['peeringDB_ix_count']>0]
    print(df)
    print(1-df.isna().sum()/df.shape[0])
    # exit()
    if OMIT_STUBS:
        df = df[df['AS_rel_degree']>1]

    ## calculate bias for all features

    # define sets of interest
    df_ris = df.loc[(df['is_ris_peer_v4']>0) | (df['is_ris_peer_v6']>0)]
    ris_asns = list(df_ris.index)
    df_rv = df.loc[df['is_routeviews_peer']>0]
    rv_asns = list(df_rv.index)
    df_atlas = df.loc[(df['nb_atlas_probes_v4']>0) | (df['nb_atlas_probes_v6']>0)]
    atlas_asns = list(df_atlas.index)

    network_sets_dict = dict()
    network_sets_dict['all'] = df
    network_sets_dict['RIPE RIS (all)'] = df_ris
    network_sets_dict['RIPE Atlas (all)'] = df_atlas
    network_sets_dict['RouteViews (all)'] = df_rv
    network_sets_dict['RIS + RV (all)'] = df.loc[(df['is_ris_peer_v4']>0) | (df['is_ris_peer_v6']>0) | (df['is_routeviews_peer']>0)]

    for m in mons:
        m_asns = list(network_sets_dict[m].index)
        for i in NB_SAMPLES:
            for j in range(NB_ITERATIONS):
                if i<len(m_asns):
                    s = random.sample(m_asns,i)
                    network_sets_dict['{}{}_{}'.format(m.split(' (')[0],i,j)] = df.loc[s]
                else:
                    network_sets_dict['{}{}_{}'.format(m.split(' (')[0],i,j)] = network_sets_dict[m]


    network_sets_dict_for_bias = {k:v[FEATURES] for k,v in network_sets_dict.items() if k != 'all'}

    params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}
    bias_df = bu.bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **params)

    # params={'method':'total_variation', 'bins':10}
    # bias_df_tv = bu.bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **params)

    # params={'method':'max_variation', 'bins':10}
    # bias_df_max = bu.bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **params)


    # print biases & save to csv
    print('Bias per monitor set (columns) and per feature (rows)')
    print_df = bias_df.copy()
    print_df.index = [n.replace('\n','') for n in FEATURE_NAMES_DICT.values()]
    print(print_df.round(2))
    print_df.round(4).to_csv(BIAS_CSV_FNAME, header=True, index=True)

    

FIG_SAVE_FNAME = './figures/Fig_bias_vs_sampling_{}.png'
FONTSIZE = 15
COLORS = ['k','b','r','m','g']
# mons = ['all', 'RIPE RIS (all)','RouteViews (all)','RIS + RV (all)','RIPE Atlas (all)']
mons = ['all', 'RIPE RIS (all)','RouteViews (all)','RIPE Atlas (all)']
def custom_plot_save_and_close(fname):
    plt.legend(fontsize=FONTSIZE, loc='upper right')
    plt.xscale('log')
    plt.axis([9,1100,0,0.30])
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlabel('#monitors', fontsize=FONTSIZE)
    plt.ylabel('mean bias', fontsize=FONTSIZE)
    plt.grid(True)
    plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.savefig(fname)
    plt.close()

h = (1.96/np.sqrt(NB_ITERATIONS))

for en, m in enumerate(mons):
    mean_bias = []
    std_bias = []
    for i in NB_SAMPLES:
        cols = [c for c in bias_df.columns if c.startswith('{}{}'.format(m.split(' (')[0],i))]
        mean_bias.append( bias_df[cols].mean().mean() )
        std_bias.append( bias_df[cols].mean().std() )
    plt.errorbar(NB_SAMPLES, mean_bias, [h*i for i in std_bias], color=COLORS[en], label=m.split(' (')[0])
    if m != 'all':
        plt.plot(NB_SAMPLES, [bias_df[m].mean()]*len(NB_SAMPLES), linestyle='--', color=COLORS[en])
    else:
        print('### Avg bias for random sampling ###')
        # print([str(nb) for nb in NB_SAMPLES])
        # print([str(nb) for nb in mean_bias])
        print('#samples: {}'.format('\t'.join([str(nb) for nb in NB_SAMPLES])))
        print('bias    : {}'.format('\t'.join([str(round(nb,2)) for nb in mean_bias])))
custom_plot_save_and_close(FIG_SAVE_FNAME.format('TOTAL'))


for ind in bias_df.index:
    for en, m in enumerate(mons):
        mean_bias = []
        std_bias = []
        for i in NB_SAMPLES:
            cols = [c for c in bias_df.columns if c.startswith('{}{}'.format(m.split(' (')[0],i))]
            mean_bias.append( bias_df.loc[ind,cols].mean() )
            std_bias.append( bias_df.loc[ind,cols].std() )
        plt.errorbar(NB_SAMPLES, mean_bias, [h*i for i in std_bias], color=COLORS[en], label=m.split(' (')[0])
        if m != 'all':
            plt.plot(NB_SAMPLES, [bias_df.loc[ind,m]]*len(NB_SAMPLES), linestyle='--', color=COLORS[en])
    custom_plot_save_and_close(FIG_SAVE_FNAME.format(ind.replace(' ','_')))


FEATURE_GROUPS = {
        'Location': ['RIR region', 'Location (country)', 'Location (continent)'],
        'Network size': ['Customer cone (#ASNs)', 'Customer cone (#prefixes)', 'Customer cone (#addresses)', 'AS hegemony'],
        'Topology': ['#neighbors (total)', '#neighbors (peers)', '#neighbors (customers)', '#neighbors (providers)'],
        'IXP related': ['#IXPs (PeeringDB)', '#facilities (PeeringDB)', 'Peering policy (PeeringDB)'],
        'Network type': ['Network type (PeeringDB)', 'Traffic ratio (PeeringDB)', 'Traffic volume (PeeringDB)', 'Scope (PeeringDB)', 'Personal ASN']
        }

for fg,ind in FEATURE_GROUPS.items():
    mean_bias = []
    std_bias = []
    h = (1.96/np.sqrt(NB_ITERATIONS))
    for en, m in enumerate(mons):
        mean_bias = []
        std_bias = []
        for i in NB_SAMPLES:
            cols = [c for c in bias_df.columns if c.startswith('{}{}'.format(m.split(' (')[0],i))]
            mean_bias.append( bias_df.loc[ind,cols].mean().mean() )
            std_bias.append( bias_df.loc[ind,cols].mean().std() )
        plt.errorbar(NB_SAMPLES, mean_bias, [h*i for i in std_bias], color=COLORS[en], label=m.split(' (')[0])
        if m != 'all':
            plt.plot(NB_SAMPLES, [bias_df.loc[ind,m].mean()]*len(NB_SAMPLES), linestyle='--', color=COLORS[en])
    custom_plot_save_and_close(FIG_SAVE_FNAME.format('group_'+fg.replace(' ','_')))



# print avg bias for infrastucture
print('### Avg bias for infrastructure ###')
print(bias_df[['RIPE RIS (all)','RouteViews (all)','RIS + RV (all)','RIPE Atlas (all)']].mean())