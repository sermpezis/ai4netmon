from ai4netmon.Analysis.aggregate_data import data_aggregation_tools as dat
from ai4netmon.Analysis.aggregate_data import graph_methods as gm
import numpy as np
import csv
from dython.nominal import associations
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pandas as pd




df = dat.load_aggregated_dataframe(preprocess=True)


transformed_features = ['AS_rank_numberAsns', 'AS_rank_numberPrefixes',
                        'AS_rank_numberAddresses', 'AS_rank_total',
                        'AS_rank_customer', 'AS_rank_peer', 'AS_rank_provider',
                        'peeringDB_ix_count', 'peeringDB_fac_count', 'cti_origin']


df[transformed_features] = df.loc[:, transformed_features].transform(lambda x: np.log(1 + x))



## ---------------------------correlation task------------------------------------

# print(df)

from dython.nominal import identify_nominal_columns

categorical_features = identify_nominal_columns(df)
# print(categorical_features)

complete_correlation = associations(df, filename= 'complete_correlation.png', figsize=(60,60))
df_complete_corr = complete_correlation['corr']
# print(df_complete_corr)
df_complete_corr.to_csv('df_complete_corr.csv')

radar_df = df[['AS_rank_iso','AS_rank_continent', 'is_personal_AS', 'peeringDB_info_scope', 'peeringDB_info_traffic', 'peeringDB_info_type','peeringDB_info_ratio','peeringDB_policy_general'
 ,'peeringDB_fac_count', 'peeringDB_ix_count', 'AS_rank_provider', 'AS_rank_customer', 'AS_rank_peer', 'AS_rank_total','AS_hegemony'
 , 'AS_rank_numberAsns', 'AS_rank_numberPrefixes', 'AS_rank_numberAddresses', 'AS_rank_source']]

radar_correlation = associations(radar_df, filename= 'radar_correlation.png', figsize=(60,60))
df_radar_corr = radar_correlation['corr']
df_radar_corr.to_csv('df_radar_corr.csv')

