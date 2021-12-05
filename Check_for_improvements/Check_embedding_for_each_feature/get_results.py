import pandas as pd
import data_preprocessing as dp


karate_club_emb_64 = ['Diff2Vec', 'NetMF', 'NodeSketch', 'Walklets', 'Node2Vec_Local', 'Node2Vec_Global',
                      'Node2Vec_wl5_global', 'Node2Vec_wl5_e3_global', 'Node2Vec_wl5_e3_local', 'bgp2vec_64',
                      'bgp2vec_32']
karate_club_emb_128 = ['Diff2Vec', 'NetMF', 'NodeSketch', 'Walklets', 'DeepWalk']
graph_emb_dimensions = 64
proceed_to_classification = True

final_df = pd.read_csv('../../Analysis/aggregate_data/final_dataframe.csv')
embeddings_df = dp.read_karateClub_embeddings_file(karate_club_emb_64[10], dimensions=graph_emb_dimensions)
embeddings_df['ASN'] = embeddings_df.ASN.astype(float)
mergedStuff = dp.merge_datasets(final_df, embeddings_df)

if proceed_to_classification:
    mergedStuff.dropna(subset=['peeringDB_info_type'], inplace=True)
    new_data = dp.change_string_class_to_categorical(mergedStuff, 'peeringDB_info_type')
    y = mergedStuff['peeringDB_info_type']
    X = mergedStuff.drop(
        ['ASN', 'AS_rank_rank', 'AS_rank_source', 'AS_rank_longitude', 'AS_rank_latitude', 'AS_rank_numberAsns',
         'AS_rank_numberPrefixes', 'AS_rank_numberAddresses', 'AS_rank_iso', 'AS_rank_total', 'AS_rank_customer',
         'AS_rank_peer', 'AS_rank_provider', 'is_personal_AS', 'peeringDB_info_ratio', 'peeringDB_info_traffic',
         'peeringDB_info_scope', 'peeringDB_info_type', 'peeringDB_info_prefixes4', 'peeringDB_info_prefixes6',
         'peeringDB_policy_general', 'peeringDB_ix_count', 'peeringDB_fac_count', 'peeringDB_created', 'AS_hegemony',
         'nb_atlas_probes_v4', 'nb_atlas_probes_v6'], axis=1)
    x_train, x_test, y_train, y_test = dp.split_data(X, y)
    dp.call_classification_models(x_train, x_test, y_train, y_test)

else:
    # Fill the NaN values with the median value of the column
    # mergedStuff['AS_hegemony'].fillna((mergedStuff['AS_hegemony'].mean()), inplace=True)
    mergedStuff.dropna(subset=['AS_rank_rank'], inplace=True)
    y = mergedStuff['AS_rank_rank']
    X = mergedStuff.drop(
        ['ASN', 'AS_rank_rank', 'AS_rank_source', 'AS_rank_longitude', 'AS_rank_latitude', 'AS_rank_numberAsns',
         'AS_rank_numberPrefixes', 'AS_rank_numberAddresses', 'AS_rank_iso', 'AS_rank_total', 'AS_rank_customer',
         'AS_rank_peer', 'AS_rank_provider', 'is_personal_AS', 'peeringDB_info_ratio', 'peeringDB_info_traffic',
         'peeringDB_info_scope', 'peeringDB_info_type', 'peeringDB_info_prefixes4', 'peeringDB_info_prefixes6',
         'peeringDB_policy_general', 'peeringDB_ix_count', 'peeringDB_fac_count', 'peeringDB_created', 'AS_hegemony',
         'nb_atlas_probes_v4', 'nb_atlas_probes_v6'], axis=1)

    x_train, x_test, y_train, y_test = dp.split_data(X, y)
    dp.call_regression_models(x_train, x_test, y_train, y_test)