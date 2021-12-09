# Datasets with aggregated data (AS rank, Peering DB, etc.) per ASN

The datasets contain a large collection of features/characteristics/data per ASN aggregated from various sources (AS rank, PeeringDB, etc.). The naming is as follows: `asn_aggregate_data_<date[YYYYMMDD]>.csv`. 
The _first column_ of the csv files corresponds to the network ASN; the data can be loaded with `pandas.read_csv(filename, header=0, index_col=0)`.

