import pandas as pd
import data_preprocessing as dp

SIMILARITY_MATRIX_FNAME = 'RIPE_RIS_similarity_embeddings_20211221.csv'
FINAL_DATAFRAME = '../../Analysis/aggregate_data/final_dataframe.csv'

df = pd.read_csv(FINAL_DATAFRAME)

with open(SIMILARITY_MATRIX_FNAME, 'r') as f:
    distance_matrix = pd.read_csv(f, header=0)
print(distance_matrix)

mergedStuff = dp.merge_datasets(df, distance_matrix)
final_dataframe = mergedStuff.drop(
    ['AS_rank_rank', 'AS_rank_source', 'AS_rank_longitude', 'AS_rank_latitude', 'AS_rank_numberAsns',
     'AS_rank_numberPrefixes', 'AS_rank_numberAddresses', 'AS_rank_total', 'AS_rank_customer',
     'AS_rank_peer', 'AS_rank_provider', 'is_personal_AS', 'peeringDB_info_ratio', 'peeringDB_info_traffic',
     'peeringDB_info_scope', 'peeringDB_info_type', 'peeringDB_info_prefixes4', 'peeringDB_info_prefixes6',
     'peeringDB_policy_general', 'peeringDB_ix_count', 'peeringDB_fac_count', 'peeringDB_created', 'AS_hegemony',
     'nb_atlas_probes_v4', 'nb_atlas_probes_v6'], axis=1)

final_dataframe['AS_rank_iso'] = dp.convert_country_to_continent(final_dataframe)
final_dataframe = final_dataframe.dropna(subset=['AS_rank_iso'])
print(final_dataframe.AS_rank_iso.value_counts())
final_dataframe.to_csv('Similarity_with_iso.csv', index=False)

