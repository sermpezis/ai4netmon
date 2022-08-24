import pandas as pd
from collections import defaultdict

from ai4netmon.Analysis.aggregate_data import data_aggregation_tools as dat
from ai4netmon.Analysis.aggregate_data import data_collectors as dc
from ai4netmon.Analysis.bias import bias_utils as bu


BIAS_CSV_FNAME = './data/bias_values_per_rrc.csv'
BIAS_CSV_FNAME_OBSERVABLE = './data/bias_values_per_rrc_observable.csv'

# load RRC 2 ASN data 
ris_peer_ip2asn, ris_peer_ip2rrc = dc.get_ripe_ris_data()
rrc2asn_dict = defaultdict(list)
for ip, rrc in ris_peer_ip2rrc.items():
    rrc2asn_dict[rrc].append( ris_peer_ip2asn[ip] )

## load data
df = dat.load_aggregated_dataframe(preprocess=True)

# calculate biases
bias_df = bu.get_bias_of_monitor_set(df=df, imp='RIPE RIS')
for rrc, rrc_asns in rrc2asn_dict.items():
    bias_df_rrc = bu.get_bias_of_monitor_set(df=df, imp=rrc, monitor_list=rrc_asns, params=None)
    bias_df = pd.concat([bias_df, bias_df_rrc], axis=1)


# save data
bias_df.to_csv(BIAS_CSV_FNAME, header=True, index=True)
bias_df.index.name = 'tag'
bias_df.to_csv(BIAS_CSV_FNAME_OBSERVABLE, header=True, index=True)