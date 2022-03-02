import pandas as pd
from ai4netmon.Analysis.bias import generate_distribution_plots as gdp
from matplotlib import pyplot as plt
from matplotlib import cm, colors


AGGREGATE_DATA_FNAME = 'https://raw.githubusercontent.com/sermpezis/ai4netmon/dev/data/aggregate_data/asn_aggregate_data_20211201.csv'
BIAS_DF = './data/bias_df__no_stubs.csv'
IMPROVEMENTS = '../../data/misc/improvements20210601.txt'
SAVE_CSV_FNAME = './data/df_bias_vs_improvement_detailed.csv'
RENAME_COLUMNS = True


## load network data
df = pd.read_csv(AGGREGATE_DATA_FNAME, header=0, index_col=0)
df['is_personal_AS'].fillna(0, inplace=True)

## load improvement data
df_imp = pd.read_csv(IMPROVEMENTS, sep=" ", names=['loc', 'IPv', 'ASN', 'improvement'])
df_imp.set_index('ASN', inplace=True)
df_imp = df_imp[(df_imp['loc']=='GLOBAL') & (df_imp['IPv']==4)]
df_imp = df_imp.loc[~df_imp.index.str.contains('{'),:]
df_imp.index = [int(i) for i in df_imp.index]
df_imp.index = df_imp.index.astype(float)
df_imp.index = df_imp.index.astype(int)

## load detailed bias data 
df_bias = pd.read_csv(BIAS_DF, header=0, index_col=0).T
df_bias.index.name = 'ASN'
ripe_bias_df = df_bias.loc['RIPE RIS']
df_bias.drop(index=['RIPE RIS'], inplace=True)
df_bias.index = df_bias.index.astype(float)
df_bias.index = df_bias.index.astype(int)
for c in df_bias.columns:
    # df_bias[c] = (df_bias[c] - ripe_bias_df[c])/ripe_bias_df[c] * 100
    df_bias[c] = df_bias[c] - ripe_bias_df[c]
df_bias = df_bias.add_prefix('bias_')


# create dataframe from existing info for plotting
df_plot = pd.merge(df_imp,df_bias, left_index=True, right_index=True)
df_plot = pd.merge(df_plot,df[['AS_rank_continent','AS_rank_iso']], how='left', left_index=True, right_index=True)
df_plot.drop(columns=['loc', 'IPv'], inplace=True)

print(df_plot)


# colors_property = 'AS_rank_continent'
# unique_values = pd.unique(df_plot[colors_property]).tolist()
# dict_colors = {v:i for i,v in enumerate(unique_values)}
# df_plot['Colors'] = [dict_colors.get(v) for v in df_plot[colors_property]]
df_plot['ASN'] = df_plot.index

if RENAME_COLUMNS:
    new_names = {
        'bias_AS_rank_source': 'RIR region', 
        'bias_AS_rank_iso': 'Country',
        'bias_AS_rank_continent': 'Continent', 
        'bias_AS_rank_numberAsns': 'Customer cone (#ASNs)',
        'bias_AS_rank_numberPrefixes': 'Customer cone (#prefixes)', 
        'bias_AS_rank_numberAddresses': 'Customer cone (#addresses)',
        'bias_AS_hegemony': 'AS hegemony', 
        'bias_AS_rank_total': '#neighbors (total)', 
        'bias_AS_rank_peer': '#neighbors (peers)', 
        'bias_AS_rank_customer': '#neighbors (customers)', 
        'bias_AS_rank_provider': '#neighbors (providers)', 
        'bias_peeringDB_ix_count': '#IXPs', 
        'bias_peeringDB_fac_count': '#facilities', 
        'bias_peeringDB_policy_general': 'Peering policy', 
        'bias_peeringDB_info_type': 'Network type', 
        'bias_peeringDB_info_ratio': 'Traffic ratio', 
        'bias_peeringDB_info_traffic': 'Traffic volume', 
        'bias_peeringDB_info_scope': 'Scope', 
        'bias_is_personal_AS': 'Personal ASN'}
    df_plot.rename(new_names, axis=1, inplace=True)
print(df_plot.columns)

df_plot.to_csv(SAVE_CSV_FNAME, index=False)

print(df_plot)