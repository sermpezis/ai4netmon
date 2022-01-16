import pandas as pd
import numpy as np
import metric_preprocessing as mp
from Analysis.embedding_similarity import cluster_similarity

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
# DO NOT KNOW IF
final_data.fillna(0, inplace=True)
