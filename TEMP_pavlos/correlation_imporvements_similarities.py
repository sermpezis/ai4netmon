import json
import numpy as np
from collections import defaultdict
import pandas as pd
from matplotlib import pyplot as plt

ASN2ASN_DIST_FNAME = './data/asn2asn__only_peers_pfx.json'
LIST_OF_RIPE_PEERS_FNAME = './list_of_RIPE_RIS_peers.json'
PATHLEN_SIMILARITIES_SUM_FNAME = 'pathlen_similarities_sum.json'
PATHLEN_SIMILARITIES_MIN_FNAME = 'pathlen_similarities_min.json'


print('Loading RIPE RIS peers dataset')
with open(LIST_OF_RIPE_PEERS_FNAME,'r') as f:
   dict_ripe = json.load(f)
print('Loading asn2asn dataset')
with open(ASN2ASN_DIST_FNAME, 'r') as f:
   asn2asn = json.load(f)
print('Loading similarities sum dataset')
with open(PATHLEN_SIMILARITIES_SUM_FNAME,'r') as f:
    total_distance = json.load(f)
print('Loading similarities min dataset')
with open(PATHLEN_SIMILARITIES_MIN_FNAME,'r') as f:
    min_distance = json.load(f)

print('Calculating improvements')
improvements = defaultdict(lambda :0)
for o_asn, dict_o_asn in asn2asn.items():
   sorted_dist = sorted(set(dict_o_asn.values()))
   list_of_min_dist_peers = [dict_ripe[p] for p,d in dict_o_asn.items() if d==sorted_dist[0]]
   if len(list_of_min_dist_peers) == 1:
      if len(sorted_dist) > 1:
         improvements[list_of_min_dist_peers[0]] += sorted_dist[1]-sorted_dist[0]

print('Calculating total similarities')
total_distance_asn = defaultdict(list)
min_distance_asn = defaultdict(list)
for ip, asn in dict_ripe.items():
    total_distance_asn[asn].append(total_distance.get(ip,np.nan))
    min_distance_asn[asn].append(min_distance.get(ip,np.nan))
avg_total_distance_asn = {asn:np.nanmean(distances) for asn,distances in total_distance_asn.items()}
min_min_distance_asn = {asn:np.nanmin(distances) for asn,distances in min_distance_asn.items()}
# avg_total_distance_asn = dict()
# for asn,distances in total_distance_asn.items():
#     print(asn)
#     print(distances)
#     avg_total_distance_asn[asn] = np.nanmean(distances)



df = pd.DataFrame()
df = pd.DataFrame.from_dict(improvements, orient='index',columns=['improvements'])
print(df)
# df = df.join(pd.DataFrame.from_dict(improvements, orient='index',columns=['improvement']))
df = df.join(pd.DataFrame.from_dict(avg_total_distance_asn, orient='index',columns=['avg_similarity']))
df = df.join(pd.DataFrame.from_dict(min_min_distance_asn, orient='index',columns=['min_similarity']))
print(df)


print('Correlation matrix')
corr = df.corr() 
print(corr)


plt.scatter(df['improvements'],df['min_similarity'])
plt.xscale('log')
plt.show()


plt.scatter(df['improvements'],df['avg_similarity'])
plt.xscale('log')
plt.show()
