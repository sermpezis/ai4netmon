import pandas as pd

df = pd.read_json('../../Datasets/PeeringDB/peeringdb_2_dump_2021_07_01.json')
data = []
keep_keys = ['info_ratio', 'info_traffic', 'info_scope', 'info_type', 'info_prefixes4',
        'info_prefixes6', 'policy_general', 'ix_count', 'fac_count', 'created']
for row in df.net['data']:
    net_row = []
    for key in keep_keys:
        if key in row:
            net_row.append(row[key])
        else:
            net_row.append(None)
    data.append(net_row)
df = pd.DataFrame(data, columns=keep_keys)
df.to_csv('PeeringDB.csv', index=False)
print(df)