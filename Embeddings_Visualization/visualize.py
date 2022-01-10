import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import json

RIPE_RIS_PEERS = '../Datasets/RIPE_RIS_peers/improvements_RIPE_RIS_peers_leave_one_out.json'
STUB_ASES = '../Analysis/remove_Stubs_from_AS_relationships/Stub_ASes.csv'

DEEPWALK_EMBEDDINGS_128 = 'StorePreprocessedEmb/PreprocessedDeepWalk128.csv'
DIFF2VEC_EMBEDDINGS_128 = 'StorePreprocessedEmb/PreprocessedDiff2Vec128.csv'
NETMF_EMBEDDINGS_128 = 'StorePreprocessedEmb/PreprocessedNetMF128.csv'
NODESKETCH_EMBEDDINGS_128 = 'StorePreprocessedEmb/PreprocessedNodeSketch128.csv'
WALKLETS_EMBEDDINGS_256 = 'StorePreprocessedEmb/PreprocessedWalklets256.csv'
NODE2VEC_EMBEDDINGS_64 = 'StorePreprocessedEmb/PreprocessedNode2Vec64.csv'
NODE2VEC_LOCAL_EMBEDDINGS_64 = 'StorePreprocessedEmb/PreprocessedNode2Vec_Local64.csv'
NODE2VEC_GLOBAL_EMBEDDINGS_64 = 'StorePreprocessedEmb/PreprocessedNode2Vec_Global64.csv'
DIFF2VEC_EMBEDDINGS_64 = 'StorePreprocessedEmb/PreprocessedDiff2Vec64.csv'
NETMF_EMBEDDINGS_64 = 'StorePreprocessedEmb/PreprocessedNetMF64.csv'
NODESKETCH_EMBEDDINGS_64 = 'StorePreprocessedEmb/PreprocessedNodeSketch64.csv'
WALKLETS_EMBEDDINGS_128 = 'StorePreprocessedEmb/PreprocessedWalklets128.csv'
NODE2VEC_WL5_E3_LOCAL = 'StorePreprocessedEmb/PreprocessedNode2Vec_wl5_e3_local64.csv'
NODE2VEC_WL5_E3_GLOBAL = 'StorePreprocessedEmb/PreprocessedNode2Vec_wl5_e3_global64.csv'
NODE2VEC_64_WL5_E1_GLOBAL = 'StorePreprocessedEmb/PreprocessedNode2Vec_wl5_global64.csv'
BGP2VEC_64 = 'StorePreprocessedEmb/Preprocessedbgp2vec64.csv'
BGP2VEC_32 = 'StorePreprocessedEmb/Preprocessedbgp2vec_32.csv'

run_script_with_All_ASES = False
choose_model = NODE2VEC_WL5_E3_GLOBAL


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
        c=[sns.color_palette()[x] for x in df_all.AS_rank_iso.map({"Asia": 0, "North America": 1, "South America": 2,
                                                                   "Europe": 3, "Africa": 4, "Oceania": 5})])

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


def read_Stub_csv(path):
    """
    :param path: Contains the path of the ASns stubs
    :return: A dataframe containing the ASns stubs
    """
    data_stubs = pd.read_csv(path)
    data_stubs.columns = ['ASN']
    data_stubs['ASN'] = data_stubs['ASN'].astype(float)
    print(len(data_stubs))

    return data_stubs


def read_embedding_model(path):
    """
    :param path: Contains the path of preprocessed graph embedding
    :return: A dataframe containing the dimensions of graph embedding, The graph embedding name
    """
    embedding_name = path
    models_name = (embedding_name.lower())
    new_data = pd.read_csv(embedding_name)
    print(len(new_data['ASN'].unique()))

    return new_data, models_name


def read_RIS_improvement_score():
    """
    :return: A dataframe containing 2 columns. The first one contains ASNs and the second one the improvement score
     that each ASN will bring to the Network
    """
    with open(RIPE_RIS_PEERS) as handle:
        dictdump = json.loads(handle.read())

    data = pd.DataFrame(dictdump.items(), columns=['ASN', 'Improvement_score'])

    data['ASN'] = data['ASN'].astype(np.int64)
    return data


if run_script_with_All_ASES:
    data, model_name = read_embedding_model(choose_model)
    df_stubs = read_Stub_csv(STUB_ASES)
    # we have drop ASes that have no info about AS_rank_iso in create_csv_for_visualization.py,
    # that's why we have this difference (42 elements missing)
    df_all = data.merge(df_stubs, how='left', indicator=True).loc[lambda x: x['_merge'] == 'left_only']
    data_emb = df_all.drop(['ASN', '_merge', 'AS_rank_iso'], axis=1)
    umap_tranformed = call_umap(data_emb)
    create_embedding_plot(umap_tranformed, df_all, model_name)
else:
    data, model_name = read_embedding_model(choose_model)
    ripe_monitors = read_RIS_improvement_score()
    df_ripe = pd.merge(ripe_monitors, data, on=['ASN'], how='left')
    # We need to drop one monitor because we do not have its embedding
    df_ripe.dropna(inplace=True)
    data_emb = df_ripe.drop(['ASN', 'AS_rank_iso', 'Improvement_score'], axis=1)
    umap_tranformed = call_umap(data_emb)
    create_embedding_plot(umap_tranformed, df_ripe, model_name)
