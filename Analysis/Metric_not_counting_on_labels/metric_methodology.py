import numpy as np
import statistics
import pandas as pd
from matplotlib import pyplot as plt
from Analysis.Metric_Impact_Hijacking import metric_preprocessing as mp
from Analysis.Metric_Impact_Hijacking import give_metric_ases_from_clusters as gmafc

run_script_with_embeddings = True
monitors_selected = 100
FINAL_DATASET = '../aggregate_data/final_dataframe.csv'

data_CAIDA = mp.read_caida_ases()
ripe_monitors = mp.read_ripe_ris_monitors()
mp.compare_ases_from_caida_ripe(data_CAIDA, ripe_monitors)

final_dataframe = pd.read_csv(FINAL_DATASET)
final_dataframe['AS_rank_iso'] = mp.convert_country_to_continent(final_dataframe)
data = mp.one_hot(final_dataframe)
data.drop(['AS_rank_iso'], axis=1, inplace=True)

if run_script_with_embeddings:
    dim = 32
    embeddings = mp.read_karateClub_embeddings_file(dim)
    final_data = mp.merge_datasets(data, embeddings)
else:
    final_data = data

final_data = final_data.drop(['AS_rank_source', 'AS_rank_longitude', 'AS_rank_latitude', 'nb_atlas_probes_v4',
                              'nb_atlas_probes_v6', 'peeringDB_created', 'peeringDB_policy_general',
                              'peeringDB_info_prefixes4', 'peeringDB_info_prefixes6', 'peeringDB_info_scope',
                              'peeringDB_info_type', 'peeringDB_info_ratio', 'peeringDB_info_traffic'], axis=1)

greedy_min, greedy_max, cluster_kmeans, cluster_spectral = gmafc.return_the_selected_monitors_from_methods()

greedy_min = [int(number) for number in greedy_min]
greedy_max = [int(number) for number in greedy_max]
cluster_kmeans = [int(number) for number in cluster_kmeans]
cluster_spectral = [int(number) for number in cluster_spectral]

data_CAIDA.fillna(0, inplace=True)
data_for_metric = []

data_caida_monitors = list(data_CAIDA.columns.values)
data_caida_monitors = data_caida_monitors[7: 295]
data_caida_monitors = [int(x) for x in data_caida_monitors]


final_list = []
for index, row in data_CAIDA.iterrows():
    new_row = str(row['impact'])
    monitors_counter = 0
    rand_monitors = []
    new_list = [new_row]

    for monitor in cluster_spectral:
        if monitor in data_caida_monitors and monitors_counter < monitors_selected:
            monitors_counter = monitors_counter + 1
            monitor_index = data_caida_monitors.index(monitor)
            rand_monitor = row.iloc[monitor_index]
            rand_monitors.append(rand_monitor)
    new_list.append(rand_monitors)

    head, tail = new_list
    if new_list != [new_row]:
        final_list.append([head] + tail)


# Create column names
initial_col = ['impact', 'observation_1']
obs_range = range(2, monitors_selected+1)
middle_cols = ['observation_' + str(i) for i in obs_range]
counter = 1
for i in middle_cols:
    initial_col.append(i)

final_df = pd.DataFrame(final_list, columns=initial_col)
final_df[initial_col] = final_df[initial_col].apply(pd.to_numeric, errors='coerce', axis=1)

for index, row in final_df.iterrows():
    final_df['estimated_impact'] = final_df.eq(1).sum(axis=1)/monitors_selected
for index, row in final_df.iterrows():
    final_df['square_error'] = pow((final_df['impact'] - final_df['estimated_impact']), 2)

for index, row in final_df.iterrows():
    rmse = np.sqrt(statistics.mean(final_df['square_error']))
print(rmse)


# fontsize = 15
# linewidth = 2
# X = list(range(1, monitors_selected))
# plt.plot(X, rmse, label="Min SpectralClustering k10", linewidth=linewidth)
# plt.xscale('log')
# plt.xlabel('#monitors', fontsize=fontsize)
# plt.ylabel('RMSE', fontsize=fontsize)
# plt.xticks(fontsize=fontsize)
# plt.yticks(fontsize=fontsize)
# plt.tight_layout()
# plt.axis([1, 100, 0.1, 0.2])
# plt.legend()
# plt.grid(True)
# plt.show()