import pandas as pd
import embedding_preprocessing as ep


karate_club_emb_64 = ['Diff2Vec', 'NetMF', 'NodeSketch', 'Walklets', 'Node2Vec_Local', 'Node2Vec_Global',
                      'Node2Vec_wl5_global', 'Node2Vec_wl5_e3_global', 'Node2Vec_wl5_e3_local', 'bgp2vec_64',
                      'bgp2vec_32']
karate_club_emb_128 = ['Diff2Vec', 'NetMF', 'NodeSketch', 'Walklets', 'DeepWalk']
graph_emb_user_choice = ''
graph_emb_dimensions = 64

df = pd.read_csv('../Analysis/aggregate_data/final_dataframe.csv')
if graph_emb_user_choice == 'Node2Vec':
    embeddings_df = ep.read_Node2Vec_embeddings_file()
    graph_embedding_model = 'Node2Vec'
else:
    if graph_emb_dimensions == 64:
        graph_embedding_model = karate_club_emb_64[6]
    elif graph_emb_dimensions == 128:
        graph_embedding_model = karate_club_emb_128[0]
    embeddings_df = ep.read_karateClub_embeddings_file(graph_embedding_model, dimensions=graph_emb_dimensions)
embeddings_df['ASN'] = embeddings_df.ASN.astype(float)
mergedStuff = ep.merge_datasets(df, embeddings_df)

final_dataframe = mergedStuff.drop(
    ['AS_rank_rank', 'AS_rank_source', 'AS_rank_longitude', 'AS_rank_latitude', 'AS_rank_numberAsns',
     'AS_rank_numberPrefixes', 'AS_rank_numberAddresses', 'AS_rank_total', 'AS_rank_customer',
     'AS_rank_peer', 'AS_rank_provider', 'is_personal_AS', 'peeringDB_info_ratio', 'peeringDB_info_traffic',
     'peeringDB_info_scope', 'peeringDB_info_type', 'peeringDB_info_prefixes4', 'peeringDB_info_prefixes6',
     'peeringDB_policy_general', 'peeringDB_ix_count', 'peeringDB_fac_count', 'peeringDB_created', 'AS_hegemony',
     'nb_atlas_probes_v4', 'nb_atlas_probes_v6'], axis=1)

final_dataframe['AS_rank_iso'] = ep.convert_country_to_continent(final_dataframe)
final_dataframe = final_dataframe.dropna(subset=['AS_rank_iso'])
print(final_dataframe.AS_rank_iso.value_counts())

final_name = ep.get_path_and_filename(graph_embedding_model, graph_emb_dimensions)
final_dataframe.to_csv(final_name, index=False)
