import pandas as pd
import numpy as np
import json
import similarity_utils as su
from ai4netmon.Analysis.aggregate_data import data_collectors as dcol


# datasets to be used
DISTANCE_MATRIX_FNAME = '../../data/similarity/ripe_ris_distance_pathlens_100k_20210701.csv'
RIPE_RIS_PEERS_FNAME = '../../data/misc/RIPE_RIS_peers_ip2asn.json'
AGGREGATE_DATA_FNAME = '../../data/aggregate_data/asn_aggregate_data_20211201.csv'

# filename to write the generated dataset
TSNE_DATASET_FNAME = '../../data/visualizations/dataset_2D_visualization_ripe_ris_distance_pathlens_100k_20210701.csv'


# load data and generate new dataset
print('Create 2D visualization dataset with rich data for RIPE RIS')
print('\t loading matrix ...')
with open(DISTANCE_MATRIX_FNAME, 'r') as f:
  distance_matrix = pd.read_csv(f, header=0, index_col=0)
similarity_matrix = su.dist_to_similarity_matrix(distance_matrix)


print('\t calculating tSNE vector ...')
df = su.similarity_matrix_to_2D_vector(similarity_matrix)


print('\t enriching with monitors data ...')
df = pd.DataFrame(df)
df.index = similarity_matrix.index
df.columns = ['X', 'Y']

print('\t\t loading RIPE RIS data ...')
with open(RIPE_RIS_PEERS_FNAME, 'r') as f:
    ripe_ris_peers = json.load(f)
df['ASN'] = [ripe_ris_peers.get(ip, np.nan) for ip in df.index]

print('\t\t loading ASN aggregate data ...')
asn_df = pd.read_csv(AGGREGATE_DATA_FNAME, header=0, index_col=0)

print('\t\t adding extra data ...')
ind_peers_with_asn_data = df['ASN'].isin(asn_df.index)
asn_peers_with_asn_data = df.loc[ind_peers_with_asn_data,'ASN']
df.loc[ind_peers_with_asn_data, 'Continent'] = asn_df.loc[asn_peers_with_asn_data, 'AS_rank_continent'].values
df.loc[ind_peers_with_asn_data, 'Type'] = asn_df.loc[asn_peers_with_asn_data, 'peeringDB_info_type'].values

print('\t\t loading RRC data ...')
ripe_ris_peer_ip2asn, ripe_ris_peer_ip2rrc = dcol.get_ripe_ris_data()
df['rrc'] = [ripe_ris_peer_ip2rrc.get(ip, np.nan) for ip in df.index]


print('\t writing to file ...')
df.to_csv(TSNE_DATASET_FNAME, index=True, index_label='Peer IP')
