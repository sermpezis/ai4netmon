import numpy as np
import pandas as pd
import json
import random
from matplotlib import pyplot as plt
from ai4netmon.Analysis.bias import bias_utils as bu
from ai4netmon.Analysis.aggregate_data import data_aggregation_tools as dat
# from ai4netmon.Analysis.bias import radar_chart
# from ai4netmon.Analysis.bias import generate_distribution_plots as gdp
import os


## datasets
ATLAS_MEASUREMENTS = './data/results__ping_probe_selection_measurements_nb100_more.json'
BIAS_CSV_FNAME = './data/bias_values_sampling_real_Atlas.csv'
FIG_SAVE_FNAME = './figures/Fig_bias_vs_sampling_real_Atlas_{}.png'

FEATURE_NAMES_DICT = bu.get_features_dict_for_visualizations() 
FEATURES = list(FEATURE_NAMES_DICT.keys())

NB_SAMPLES = [10, 20, 50, 100]
NB_ITERATIONS = 100

mons = ['all','Atlas (random)']
colors_infra = {'all':'k', 'Atlas (random)':'g', 'Atlas (platform)':'g'}
marker_infra = {'all':None, 'Atlas (random)':None, 'Atlas (platform)':'o'}
FONTSIZE = 15


if os.path.exists(BIAS_CSV_FNAME):
    bias_df = pd.read_csv(BIAS_CSV_FNAME, header=0, index_col=0)
else:
    ## load data
    df = dat.load_aggregated_dataframe(preprocess=True)
    df_atlas = df.loc[(df['nb_atlas_probes_v4']>0) | (df['nb_atlas_probes_v6']>0)]
    atlas_asns = list(df_atlas.index)

    network_sets_dict = dict()
    network_sets_dict['all'] = df
    network_sets_dict['Atlas (random)'] = df_atlas
    
    for m in mons:
        m_asns = list(network_sets_dict[m].index)
        for i in NB_SAMPLES:
            for j in range(NB_ITERATIONS):
                if i<len(m_asns):
                    s = random.sample(m_asns,i)
                    network_sets_dict['{}{}_{}'.format(m,i,j)] = df.loc[s]
                else:
                    network_sets_dict['{}{}_{}'.format(m,i,j)] = network_sets_dict[m]

    with open(ATLAS_MEASUREMENTS, 'r') as f:
        measurements_dict = json.load(f)

    for i in NB_SAMPLES:
        j = 0
        lens = []
        for m, md in measurements_dict.items():
            m_asns = [k for k in md['asns_v4'] if k in df.index]
            if i<len(m_asns):
                m_df = df.loc[m_asns[0:i]]
            else:
                m_df = df.loc[m_asns]
            network_sets_dict['Atlas (platform){}_{}'.format(i,j)] = m_df
            lens.append(len(set(m_df.index)))
            j += 1
        import numpy as np
        print(i, np.mean(lens))


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

    
def custom_plot_save_and_close(fname):
    plt.legend(fontsize=FONTSIZE, loc='upper right')
    plt.xscale('linear')
    plt.axis([5,105,0,0.40])
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlabel('#monitors', fontsize=FONTSIZE)
    plt.ylabel('avg. bias score', fontsize=FONTSIZE)
    plt.grid(True)
    plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.savefig(fname)
    plt.close()

h = (1.96/np.sqrt(NB_ITERATIONS))

mons = ['all', 'Atlas (random)','Atlas (platform)']
for m in mons:
    mean_bias = []
    std_bias = []
    for i in NB_SAMPLES:
        cols = [c for c in bias_df.columns if c.startswith('{}{}_'.format(m,i))]
        mean_bias.append( bias_df[cols].mean().mean() )
        std_bias.append( bias_df[cols].mean().std() )
    plt.errorbar(NB_SAMPLES, mean_bias, [h*i for i in std_bias], color=colors_infra[m], marker=marker_infra[m], label=m)
    if m not in ['all', 'Atlas (platform)']:
        plt.plot(NB_SAMPLES, [bias_df[m].mean()]*len(NB_SAMPLES), linestyle='--', color=colors_infra[m])
    else:
        print('### Avg bias for random sampling ###')
        print('#samples: {}'.format('\t'.join([str(nb) for nb in NB_SAMPLES])))
        print('bias    : {}'.format('\t'.join([str(round(nb,2)) for nb in mean_bias])))
custom_plot_save_and_close(FIG_SAVE_FNAME.format('TOTAL'))