import data_preprocessing as dp
import call_ML_models as cmm
import numpy as np

karate_club_emb_64 = ['Diff2Vec', 'NetMF', 'NodeSketch', 'Walklets', 'Node2Vec_Local', 'Node2Vec_Global']
karate_club_emb_128 = ['Diff2Vec', 'NetMF', 'NodeSketch', 'Walklets', 'DeepWalk']
graph_emb_user_choice = ''
graph_emb_dimensions = 128

improvement_df = dp.read_RIS_improvement_score()
if graph_emb_user_choice == 'Node2Vec':
    embeddings_df = dp.read_Node2Vec_embeddings_file()
else:
    if graph_emb_dimensions == 64:
        embeddings_df = dp.read_karateClub_embeddings_file(karate_club_emb_64[3], dimensions=graph_emb_dimensions)
    elif graph_emb_dimensions == 128:
        embeddings_df = dp.read_karateClub_embeddings_file(karate_club_emb_128[1], dimensions=graph_emb_dimensions)
mergedStuff = dp.merge_datasets(improvement_df, embeddings_df)

y = mergedStuff['Improvement_score']
X = mergedStuff.drop(['Improvement_score', 'ASN'], axis=1)
x_train_pca, x_test_pca, y_train_pca, y_test_pca = dp.split_data_with_pca(X, np.log(y))
x_train, x_test, y_train, y_test = dp.split_data(X, y)
cmm.call_methods(x_train, x_test, y_train, y_test, x_train_pca, x_test_pca, y_train_pca, y_test_pca)
