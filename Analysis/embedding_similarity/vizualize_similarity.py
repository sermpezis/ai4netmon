import umap.umap_ as umap
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

SIMILARITY_MATRIX_FNAME = 'Similarity_with_iso.csv'


def create_embedding_plot(two_dimensions, model_name):
    """
    Give us the final plot
    :param embedding: The transformed data in 2-dimensions
    :param df_all: The merged dataframe
    :param model_name: The model's name
    """
    new = two_dimensions.drop('AS_rank_iso', axis=1)
    scatter = plt.scatter(
        new.iloc[:, 0],
        new.iloc[:, 1],
        s=10,
        c=[sns.color_palette()[x] for x in two_dimensions.AS_rank_iso.map({"Asia": 0, "North America": 1, "South America": 2,
                                                                           "Europe": 3, "Africa": 4, "Oceania": 5})])

    pops = []
    for i in range(0, len(two_dimensions.AS_rank_iso.unique())):
        x, y, z = sns.color_palette()[i]
        pops.append(mpatches.Patch(color='#{:02x}{:02x}{:02x}'.format(int(x * 255), int(y * 255), int(z * 255)),
                                   label=two_dimensions.AS_rank_iso.unique()[i]))

    plt.legend(handles=pops)
    plt.title('UMAP projection of the ' + str(model_name), fontsize=12)
    plt.savefig(str(model_name) + f'.png')
    plt.show()


def call_umap(embed):
    """
    :param embed: The merged dataframe without columns=['ASN', '_merge', 'AS_rank_iso']
    :return: Data transformed in 2-dimensions
    """
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(embed)

    return embedding


with open(SIMILARITY_MATRIX_FNAME, 'r') as f:
    similarity_matrix = pd.read_csv(f, header=0)
print(similarity_matrix)

data_similarity = similarity_matrix.drop(['ASN', 'AS_rank_iso'], axis=1)
model_name = 'BGP2VEC_Global'

similarity_matrix = similarity_matrix[['AS_rank_iso']]
umap_tranformed = call_umap(data_similarity)
umap_df = pd.DataFrame(umap_tranformed, columns=['X_axes', 'Y_axes'])

new_data = umap_df.join(similarity_matrix)
print(new_data['AS_rank_iso'].value_counts())
create_embedding_plot(new_data, model_name)
