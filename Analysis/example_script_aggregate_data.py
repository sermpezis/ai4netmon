import pandas as pd

# load existing file
AGGREGATE_DATA_FNAME = '../data/aggregate_data/asn_aggregate_data_20211201.csv'

df = pd.read_csv(AGGREGATE_DATA_FNAME, header=0, index_col=0)
print(df)
print(df.columns)


# generate file
from ai4netmon.Analysis.aggregate_data import data_aggregation_tools as dat

ALL_DATASETS = ['AS_rank', 'personal', 'PeeringDB', 'AS_hegemony', 'Atlas_probes', 'RIPE_RIS', 'RouteViews']
df = dat.create_dataframe_from_multiple_datasets(ALL_DATASETS)

print(df)
print(df.columns)
