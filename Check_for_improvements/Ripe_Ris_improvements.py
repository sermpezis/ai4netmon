import data_preprocessing as dp
import call_ML_models as cmm
import numpy as np

karate_club_emb_64 = ['Diff2Vec', 'NetMF', 'NodeSketch', 'Walklets', 'Node2Vec_Local', 'Node2Vec_Global']
karate_club_emb_128 = ['Diff2Vec', 'NetMF', 'NodeSketch', 'Walklets', 'DeepWalk']
graph_emb_user_choice = ''
run_script_with_Stubs = False
graph_emb_dimensions = 64

improvement_df = dp.read_RIS_improvement_score()
if graph_emb_user_choice == 'Node2Vec':
    embeddings_df = dp.read_Node2Vec_embeddings_file()
else:
    if graph_emb_dimensions == 64:
        embeddings_df = dp.read_karateClub_embeddings_file(karate_club_emb_64[0], dimensions=graph_emb_dimensions)
    elif graph_emb_dimensions == 128:
        embeddings_df = dp.read_karateClub_embeddings_file(karate_club_emb_128[3], dimensions=graph_emb_dimensions)
mergedStuff = dp.merge_datasets(improvement_df, embeddings_df)

if run_script_with_Stubs:
    mergedStuff = dp.merge_datasets(improvement_df, embeddings_df)
    final = mergedStuff
else:
    mergedStuff = dp.merge_datasets(improvement_df, embeddings_df)
    data_stubs = dp.read_df_of_Stub_ASes()
    df_without_Stubs = mergedStuff.merge(data_stubs, how='left', indicator=True).loc[lambda x: x['_merge'] == 'left_only']
    df_without_Stubs.drop(['_merge'], axis=1, inplace=True)
    final = df_without_Stubs

y = final['Improvement_score']
X = final.drop(['Improvement_score', 'ASN'], axis=1)
x_train_pca, x_test_pca, y_train_pca, y_test_pca = dp.split_data_with_pca(X, np.log(y))
x_train, x_test, y_train, y_test = dp.split_data(X, y)
cmm.call_methods(x_train, x_test, y_train, y_test, x_train_pca, x_test_pca, y_train_pca, y_test_pca)
