import umap.umap_ as umap
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

SIMILARITY_MATRIX_FNAME = 'Similarity_with_iso.csv'


def create_embedding_plot(two_dimensions, df_all, model_name):
    """
    Give us the final plot
    :param embedding: The transformed data in 2-dimensions
    :param df_all: The merged dataframe
    :param model_name: The model's name
    """
    scatter = plt.scatter(
        two_dimensions[:, 0],
        two_dimensions[:, 1],
        s=10,
        c=[sns.color_palette()[x] for x in df_all.AS_rank_iso.map({"Europe": 0, "North America": 1, "Asia": 2,
                                                                   "South America": 3, "Oceania": 4, "Africa": 5})])

    pops = []
    for i in range(0, len(df_all.AS_rank_iso.unique())):
        x, y, z = sns.color_palette()[i]
        pops.append(mpatches.Patch(color='#{:02x}{:02x}{:02x}'.format(int(x * 255), int(y * 255), int(z * 255)),
                                   label=df_all.AS_rank_iso.unique()[i]))

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

data_similarity = similarity_matrix.drop(['ASN', 'AS_rank_iso'], axis=1)
model_name = 'Node2Vec_Global'
umap_tranformed = call_umap(data_similarity)
create_embedding_plot(umap_tranformed, similarity_matrix, model_name)
