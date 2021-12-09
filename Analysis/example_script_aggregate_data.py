import pandas as pd

AGGREGATE_DATA_FNAME = '../data/aggregate_data/asn_aggregate_data_20211201.csv'

df = pd.read_csv(AGGREGATE_DATA_FNAME, header=0, index_col=0)
print(df)
print(df.columns)