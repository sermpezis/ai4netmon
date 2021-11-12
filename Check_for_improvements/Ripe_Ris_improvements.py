import data_preprocessing as dp
import call_ML_models as cmm
import numpy as np

karate_club_emb = ['Diff2Vec', 'NetMF', 'NodeSketch', 'Walklets']

improvement_df = dp.read_RIS_improvement_score()
# embeddings_df = dp.read_Node2Vec_embeddings_file()
embeddings_df = dp.read_karateClub_embeddings_file(karate_club_emb[3])

mergedStuff = dp.merge_datasets(improvement_df, embeddings_df)

y = mergedStuff['Improvement_score']
X = mergedStuff.drop(['Improvement_score', 'ASN'], axis=1)
x_train_pca, x_test_pca, y_train_pca, y_test_pca = dp.split_data_with_pca(X, np.log(y))
x_train, x_test, y_train, y_test = dp.split_data(X, np.log(y))
cmm.call_methods(x_train, x_test, y_train, y_test, x_train_pca, x_test_pca, y_train_pca, y_test_pca)