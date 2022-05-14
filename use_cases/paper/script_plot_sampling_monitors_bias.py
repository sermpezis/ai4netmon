import numpy as np
import pandas as pd
import json
import random
from matplotlib import pyplot as plt
from ai4netmon.Analysis.bias import bias_utils as bu
from ai4netmon.Analysis.aggregate_data import data_aggregation_tools as dat
from ai4netmon.Analysis.bias import radar_chart
import os


## filenames & parameters
BIAS_CSV_FNAME = './data/bias_values_random_samples.csv'
FIG_SAVE_FNAME = './figures/Fig_bias_vs_sampling_{}.png'
FIG_RADAR_FNAME_FORMAT = './figures/Fig_radar_sampling_{}.png'

NB_SAMPLES = [10, 20, 50, 100, 200, 500, 1000]
NB_ITERATIONS = 100

mons = ['all', 'RIS','RV','RIS & RV','Atlas']
colors_infra = {'all':'k', 'RIS':'b', 'RV':'r', 'Atlas':'g', 'RIS & RV':'m'}
FONTSIZE = 15
FONTSIZE_SMALL = 12

FEATURE_NAMES_DICT = bu.get_features_dict_for_visualizations() 
FEATURES = list(FEATURE_NAMES_DICT.keys())


if os.path.exists(BIAS_CSV_FNAME):
    bias_df = pd.read_csv(BIAS_CSV_FNAME, header=0, index_col=0)
else:
    ## load data
    df = dat.load_aggregated_dataframe(preprocess=True)
    
    ## calculate bias for all features
    # define sets of interest
    df_ris = df.loc[(df['is_ris_peer_v4']>0) | (df['is_ris_peer_v6']>0)]
    ris_asns = list(df_ris.index)
    df_rv = df.loc[df['is_routeviews_peer']>0]
    rv_asns = list(df_rv.index)
    df_atlas = df.loc[(df['nb_atlas_probes_v4']>0) | (df['nb_atlas_probes_v6']>0)]
    atlas_asns = list(df_atlas.index)
    df_ris_rv = df.loc[(df['is_ris_peer_v4']>0) | (df['is_ris_peer_v6']>0) | (df['is_routeviews_peer']>0)]
    ris_rv_asns = list(df_ris_rv.index)

    ## create dicts for input to bias method
    network_sets_dict = dict()
    network_sets_dict['all'] = df
    network_sets_dict['Atlas'] = df_atlas
    network_sets_dict['RIS'] = df_ris
    network_sets_dict['RV'] = df_rv
    network_sets_dict['RIS & RV'] = df_ris_rv

    for m in mons:
        print(m)
        m_asns = list(network_sets_dict[m].index)
        for i in NB_SAMPLES:
            for j in range(NB_ITERATIONS):
                print('samples {}, iter {}/{} \r'.format(i,j,NB_ITERATIONS), end=' ')
                if i<len(m_asns):
                    s = random.sample(m_asns,i)
                    network_sets_dict['{}{}_{}'.format(m,i,j)] = df.loc[s]
                else:
                    network_sets_dict['{}{}_{}'.format(m,i,j)] = network_sets_dict[m]


    network_sets_dict_for_bias = {k:v[FEATURES] for k,v in network_sets_dict.items() if k != 'all'}

    params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}
    bias_df = bu.bias_score_dataframe(df[FEATURES], network_sets_dict_for_bias, **params)

    # print biases & save to csv
    print('Bias per monitor set (columns) and per feature (rows)')
    print_df = bias_df.copy()
    print_df.index = [n.replace('\n','') for n in FEATURE_NAMES_DICT.values()]
    print(print_df.round(2))
    print_df.round(4).to_csv(BIAS_CSV_FNAME, header=True, index=True)

    

def custom_plot_save_and_close(fname):
    plt.legend(fontsize=FONTSIZE, loc='upper right')
    plt.xscale('log')
    plt.axis([9,1100,0,0.40])
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlabel('#monitors', fontsize=FONTSIZE)
    plt.ylabel('avg. bias score', fontsize=FONTSIZE)
    plt.grid(True)
    plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.savefig(fname)
    plt.close()



## plot bias vs nb monitors
plot_mons = ['all', 'Atlas', 'RIS', 'RV']
h = (1.96/np.sqrt(NB_ITERATIONS))
for m in plot_mons:
    mean_bias = []
    std_bias = []
    for i in NB_SAMPLES:
        cols = [c for c in bias_df.columns if c.startswith('{}{}_'.format(m,i))]
        mean_bias.append( bias_df[cols].mean().mean() )
        std_bias.append( bias_df[cols].mean().std() )
    plt.errorbar(NB_SAMPLES, mean_bias, [h*i for i in std_bias], color=colors_infra[m], label=m)
    if m != 'all':
        plt.plot(NB_SAMPLES, [bias_df[m].mean()]*len(NB_SAMPLES), linestyle='--', color=colors_infra[m])
    else:
        print('### Avg bias for random sampling ###')
        # print([str(nb) for nb in NB_SAMPLES])
        # print([str(nb) for nb in mean_bias])
        print('#samples: {}'.format('\t'.join([str(nb) for nb in NB_SAMPLES])))
        print('bias    : {}'.format('\t'.join([str(round(nb,2)) for nb in mean_bias])))
custom_plot_save_and_close(FIG_SAVE_FNAME.format('TOTAL'))


## print avg bias for infrastucture
print('### Avg bias for infrastructure ###')
print(bias_df[['Atlas','RIS','RV','RIS & RV']].mean())



## plot radar plot for different sample sizes
NB_SAMPLES_PLOT = [10, 20, 100]
plot_mons = ['Atlas', 'RIS', 'RV']
for m in plot_mons:
    bias_df_avg = pd.DataFrame()
    for nb in NB_SAMPLES_PLOT:
        cols = [c for c in bias_df.columns if c.startswith('{}{}_'.format(m,nb))]
        bias_df_avg['{} ({} samples)'.format(m,nb)] = bias_df[cols].mean(axis=1)
    bias_df_avg[m] = bias_df[[m]]
    radar_chart.plot_radar_from_dataframe(bias_df_avg, colors=None, frame='polygon', 
        legend_loc=(0.75, .95), fontsize=FONTSIZE, fontsize_features=FONTSIZE_SMALL,
        save_filename=FIG_RADAR_FNAME_FORMAT.format(m), varlabels={f.replace('\n',''):f for f in FEATURE_NAMES_DICT.values()})


