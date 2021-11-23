import pandas as pd

data = pd.read_csv("../../Datasets/AS-relationships/20210701.as-rel2.txt", sep="|", skiprows=180)
data.columns = ['node1', 'node2', 'link', 'protocol']
data.drop(['protocol'], axis=1, inplace=True)

df = pd.DataFrame(data['node1'].value_counts())
df.index.name = 'node1'
df.columns = ['count']
df['node1'] = df.index

list_of_stubs = []
for index, row in df.iterrows():
    if row[0] == 1:
        list_of_stubs.append(row[1])

print(len(list_of_stubs))
stubs_series = pd.Series((list_of_stubs))
stubs_series.columns = ['Stub_ASes']
print(stubs_series)
stubs_series.to_csv('Stub_ASes.csv', index=False, header=['Stub_ASes'])


