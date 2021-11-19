import pandas as pd
import matplotlib.patches as mpatches
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns

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

embedding_name = DIFF2VEC_EMBEDDINGS_128
model_name = (embedding_name.lower())
data = pd.read_csv(embedding_name)
data_emb = data.drop(['AS_rank_iso', 'ASN'], axis=1)

reducer = umap.UMAP()
embedding = reducer.fit_transform(data_emb)

scatter = plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=[sns.color_palette()[x] for x in data.AS_rank_iso.map({"Europe": 0, "North America": 1, "Asia": 2,
                                                                   "South America": 3, "Oceania": 4, "Africa": 5})])

pops = []
for i in range(0, len(data.AS_rank_iso.unique())):
    x, y, z = sns.color_palette()[i]
    pops.append(mpatches.Patch(color='#{:02x}{:02x}{:02x}'.format(int(x*255), int(y*255), int(z*255)), label=data.AS_rank_iso.unique()[i]))

plt.legend(handles=pops)
plt.title('UMAP projection of the ' + str(model_name), fontsize=12)
plt.show()
