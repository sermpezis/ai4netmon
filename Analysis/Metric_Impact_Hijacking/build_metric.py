import pandas as pd
import numpy as np
import random
import metric_preprocessing as mp
import give_metric_ases_from_clusters as gmafc

run_script_with_embeddings = True

data_CAIDA = mp.read_caida_ases()
ripe_monitors = mp.read_ripe_ris_monitors()
mp.compare_ases_from_caida_ripe(data_CAIDA, ripe_monitors)

final_dataframe = pd.read_csv('../aggregate_data/final_dataframe.csv')

final_dataframe['AS_rank_iso'] = mp.convert_country_to_continent(final_dataframe)
data = mp.one_hot(final_dataframe)
data.drop(['AS_rank_iso'], axis=1, inplace=True)

if run_script_with_embeddings:
    dim = 32
    embeddings = mp.read_karateClub_embeddings_file(dim)
    final_data = mp.merge_datasets(data, embeddings)
else:
    final_data = data

# need to drop elements
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
print(data_caida_monitors)

final_list = []
for index, row in data_CAIDA.iterrows():
    new_row = str(row['impact'])
    monitors_counter = 0
    rand_monitors = []
    new_list = [new_row]
    #  ++ θα πρέπει να έχω και μια μεταβλητή (monitors_counter) που όταν βρει 50 ίδια να σταματάει
    greedy_min = greedy_min[:100]

    for monitor in greedy_min:
        if monitor in data_caida_monitors and monitors_counter < 10:
            monitors_counter = monitors_counter + 1
            monitor_index = data_caida_monitors.index(monitor)
            rand_monitor = row.iloc[monitor_index]
            rand_monitors.append(rand_monitor)
    new_list.append(rand_monitors)

    head, tail = new_list
    if new_list != [new_row]:
        final_list.append([head] + tail)
    print(final_list)


# Create column names
initial_col = ['impact', 'observation_1']
obs_range = range(2, 11)
middle_cols = ['observation_' + str(i) for i in obs_range]
counter = 1
for i in middle_cols:
    initial_col.append(i)

final_df = pd.DataFrame(final_list, columns=initial_col)
print(final_df)