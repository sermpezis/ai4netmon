from ai4netmon.Analysis.bias import bias_utils as bu
from ai4netmon.Analysis.bias import radar_chart
from ai4netmon.Analysis.aggregate_data import data_aggregation_tools as dat
from ai4netmon.Analysis.bias import generate_distribution_plots as gdp
from matplotlib import pyplot as plt
import numpy as np


## set variables
FIG_RADAR_FNAME_FORMAT = './figures/fig_radar_{}.png'
SAVE_PLOTS_DISTRIBUTION_FNAME_FORMAT = './figures/Fig_{}_{}'

## select features for visualization
FEATURE_NAMES_DICT = bu.get_features_dict_for_visualizations() 
FEATURES = list(FEATURE_NAMES_DICT.keys())

## load data
df = dat.load_aggregated_dataframe(preprocess=True)

## select infrastructure
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
network_sets_dict['RIPE Atlas'] = df_atlas
network_sets_dict['RIPE RIS'] = df_ris
network_sets_dict['RouteViews'] = df_rv


# avg_nb_neighbors = []
width = 0.35  # the width of the bars
x_loc = [i*width for i in [-1.5, -0.5, 0.5, 1.5]]
i = 0
for k,d in network_sets_dict.items():
    print(i)
    print(k)
    # avg_nb_neighbors.append()
    plt.bar(x_loc[i], np.nanmean(d['AS_rank_total']), width, label=k)
    i+=1

FONTSIZE = 15
plt.axis([-3*width, 3*width, 0, 800])
plt.grid(True)
plt.legend(fontsize=FONTSIZE)
plt.ylabel('avg. #neighbors per AS', fontsize=FONTSIZE)
plt.xticks([])
plt.yticks(fontsize=FONTSIZE)
plt.subplots_adjust(left=0.15, bottom=0.15)
plt.xticks()
plt.savefig('test_bar.png')