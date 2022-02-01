import data_preprocessing as dp
import call_ML_models as cmm
import numpy as np
import warnings

warnings.filterwarnings('ignore')

karate_club_emb_64 = ['Diff2Vec', 'NetMF', 'NodeSketch', 'Walklets', 'Node2Vec_Local', 'Node2Vec_Global',
                      'Node2Vec_wl5_global', 'Node2Vec_wl5_e3_global', 'Node2Vec_wl5_e3_local', 'bgp2vec_64',
                      'bgp2vec_32']
karate_club_emb_128 = ['Diff2Vec', 'NetMF', 'NodeSketch', 'Walklets', 'DeepWalk']
graph_emb_user_choice = ''
run_script_with_Stubs = False
z_score = False
inner_outer_fences = False
run_only_RIPE_RIS = False
graph_emb_dimensions = 64

if run_only_RIPE_RIS:
    improvement_df = dp.read_RIS_improvement_score()
else:
    improvement_df = dp.read_improvement_score()

if graph_emb_user_choice == 'Node2Vec':
    embeddings_df = dp.read_Node2Vec_embeddings_file()
else:
    if graph_emb_dimensions == 64:
        embeddings_df = dp.read_karateClub_embeddings_file(karate_club_emb_64[4], dimensions=graph_emb_dimensions)
    elif graph_emb_dimensions == 128:
        embeddings_df = dp.read_karateClub_embeddings_file(karate_club_emb_128[3], dimensions=graph_emb_dimensions)
final_df = dp.read_final_dataset()

if run_script_with_Stubs:
    mergedStuff = dp.merge_datasets(improvement_df, embeddings_df, final_df, z_score, inner_outer_fences)
    final = mergedStuff
else:
    mergedStuff = dp.merge_datasets(improvement_df, embeddings_df, final_df, z_score, inner_outer_fences)
    data_stubs = dp.read_df_of_Stub_ASes()
    Stubs = mergedStuff.merge(data_stubs, how='inner', indicator=True)
    print(Stubs)
    df_without_Stubs = mergedStuff.merge(data_stubs, how='left', indicator=True).loc[
        lambda x: x['_merge'] == 'left_only']
    df_without_Stubs.drop(['_merge'], axis=1, inplace=True)
    final = df_without_Stubs

if z_score:
    y = final['Improvement_score']
    X = final.drop(['ASN', 'AS_rank_rank', 'AS_rank_longitude', 'AS_rank_latitude',
                    'nb_atlas_probes_v4', 'nb_atlas_probes_v6', 'Improvement_score'], axis=1)
else:
    final = dp.convert_string_to_categorical(final)
    y = final['Improvement_score']
    X = final.drop(['ASN', 'AS_rank_rank', 'AS_rank_source', 'AS_rank_longitude', 'AS_rank_latitude', 'AS_rank_iso',
                    'nb_atlas_probes_v4', 'nb_atlas_probes_v6', 'peeringDB_created', 'Improvement_score'], axis=1)


flag_k_fold = False
if flag_k_fold:
    dp.split_data_with_pca(X, np.log(y), flag_k_fold)
    cmm.clear_lists()
    dp.split_data(X, y, flag_k_fold)
else:
    x_train_pca, x_test_pca, y_train_pca, y_test_pca = dp.split_data_with_pca(X, np.log(y), flag_k_fold)
    x_train, x_test, y_train, y_test = dp.split_data(X, y, flag_k_fold)
    cmm.call_methods(x_train, x_test, y_train, y_test, x_train_pca, x_test_pca, y_train_pca, y_test_pca)

# Do NOT run SMOTE with stratify. IF you want to run SMOTE, delete stratify=y_binned in split_data and split_data_with_pca
Synthetic_Minority_Over_Sampling_Technique = False
if Synthetic_Minority_Over_Sampling_Technique:
    dp.call_smogn(final, flag_k_fold)
