import pandas as pd
from ai4netmon.Analysis.aggregate_data import data_aggregation_tools as dat
import json 

## datasets
df = dat.load_aggregated_dataframe(preprocess=True)
df.index = df.index.astype(str,copy=False)

set_infra = dict()
set_infra['RIS'] = df.loc[(df['is_ris_peer_v4']>0) | (df['is_ris_peer_v6']>0)]
set_infra['Atlas'] = df.loc[(df['nb_atlas_probes_v4']>0) | (df['nb_atlas_probes_v6']>0)]
set_infra['RV'] = df.loc[df['is_routeviews_peer']>0]

with open('List_of_ASNs_infrastructure.json', 'w') as f:
	json.dump({k:[int(float(i)) for i in list(v.index)] for k,v in set_infra.items()},f)