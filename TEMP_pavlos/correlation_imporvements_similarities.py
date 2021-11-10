import json
import numpy as np
from collections import defaultdict
import pandas as pd
from matplotlib import pyplot as plt
import os.path


ASN2ASN_DIST_FNAME = './data/asn2asn__only_peers_pfx.json'
LIST_OF_RIPE_PEERS_FNAME = './list_of_RIPE_RIS_peers.json'
PATHLEN_DISTANCES_SUM_FNAME = './data/pathlen_distances_sum.json'
PATHLEN_DISTANCES_AVG_FNAME = './data/pathlen_distances_avg.json'
PATHLEN_DISTANCES_MIN_FNAME = './data/pathlen_distances_min.json'
PATHLEN_DISTANCES_MAX_FNAME = './data/pathlen_distances_max.json'
PATHLEN_DISTANCES_SUM_5MAX_FNAME = './data/pathlen_distances_sum_5max.json'
PATHLEN_DISTANCES_SUM_10MAX_FNAME = './data/pathlen_distances_sum_10max.json'
PATHLEN_DISTANCES_SUM_5MIN_FNAME = './data/pathlen_distances_sum_5min.json'
PATHLEN_DISTANCES_SUM_10MIN_FNAME = './data/pathlen_distances_sum_10min.json'
IMPROVEMENTS_PER_PEER_LEAVE_ONE_OUT_FNAME = './data/improvements_RIPE_RIS_peers_leave_one_out.json'



print('Loading RIPE RIS peers dataset')
with open(LIST_OF_RIPE_PEERS_FNAME,'r') as f:
   dict_ripe = json.load(f)

print('Loading similarities sum dataset')
with open(PATHLEN_SIMILARITIES_SUM_FNAME,'r') as f:
    sum_distance = json.load(f)

print('Loading similarities avg dataset')
with open(PATHLEN_SIMILARITIES_AVG_FNAME,'r') as f:
    avg_distance = json.load(f)

print('Loading similarities min dataset')
with open(PATHLEN_SIMILARITIES_MIN_FNAME,'r') as f:
    min_distance = json.load(f)

print('Loading similarities max dataset')
with open(PATHLEN_SIMILARITIES_MAX_FNAME,'r') as f:
    max_distance = json.load(f)


if not os.path.isfile(IMPROVEMENTS_PER_PEER_LEAVE_ONE_OUT_FNAME):
   print('Loading asn2asn dataset')
   with open(ASN2ASN_DIST_FNAME, 'r') as f:
      asn2asn = json.load(f)

   print('Calculating improvements')
   improvements = defaultdict(lambda :0)
   for o_asn, dict_o_asn in asn2asn.items():
      sorted_dist = sorted(set(dict_o_asn.values()))
      list_of_min_dist_peers = [dict_ripe[p] for p,d in dict_o_asn.items() if d==sorted_dist[0]]
      if len(list_of_min_dist_peers) == 1:
         if len(sorted_dist) > 1:
            improvements[list_of_min_dist_peers[0]] += sorted_dist[1]-sorted_dist[0]

   print('Saving improvements dataset')
   with open(IMPROVEMENTS_PER_PEER_LEAVE_ONE_OUT_FNAME, 'w') as f:
      json.dump(improvements,f)
else:
   print('Loading improvements dataset')
   with open(IMPROVEMENTS_PER_PEER_LEAVE_ONE_OUT_FNAME, 'r') as f:
      improvements_str = json.load(f)
      improvements = {int(k):v for k,v in improvements_str.items()}

print('Calculating total similarities')
sum_distance_asn = defaultdict(list)
avg_distance_asn = defaultdict(list)
min_distance_asn = defaultdict(list)
max_distance_asn = defaultdict(list)
for ip, asn in dict_ripe.items():
   sum_distance_asn[asn].append(sum_distance.get(ip,np.nan))
   avg_distance_asn[asn].append(avg_distance.get(ip,np.nan))
   min_distance_asn[asn].append(min_distance.get(ip,np.nan))
   max_distance_asn[asn].append(max_distance.get(ip,np.nan))
avg_sum_distance_asn = {asn:np.nanmean(distances) for asn,distances in sum_distance_asn.items()}
avg_avg_distance_asn = {asn:np.nanmean(distances) for asn,distances in avg_distance_asn.items()}
min_min_distance_asn = {asn:np.nanmin(distances) for asn,distances in min_distance_asn.items()}
max_max_distance_asn = {asn:np.nanmax(distances) for asn,distances in max_distance_asn.items()}

# avg_total_distance_asn = dict()
# for asn,distances in total_distance_asn.items():
#     print(asn)
#     print(distances)
#     avg_total_distance_asn[asn] = np.nanmean(distances)



df = pd.DataFrame()
df = pd.DataFrame.from_dict(improvements, orient='index',columns=['improvements'])
df = df.join(pd.DataFrame.from_dict(avg_sum_distance_asn, orient='index',columns=['sum_distance']))
df = df.join(pd.DataFrame.from_dict(avg_avg_distance_asn, orient='index',columns=['avg_distance']))
df = df.join(pd.DataFrame.from_dict(min_min_distance_asn, orient='index',columns=['min_distance']))
df = df.join(pd.DataFrame.from_dict(max_max_distance_asn, orient='index',columns=['max_distance']))
df['sum_similarity'] = 1/df['sum_distance']
df['avg_similarity'] = 1/df['avg_distance']
df['min_similarity'] = 1/df['max_distance']
df['max_similarity'] = 1/df['min_distance']
df.replace(np.inf,np.nan,inplace=True)
print(df)


print('Correlation matrix')
corr = df.corr() 
print(corr)


plt.scatter(df['improvements'],df['min_distance'])
plt.xscale('log')
plt.savefig('fig_scatter_improvements_vs_min_distance.png')
plt.show()

plt.scatter(df['improvements'],df['max_distance'])
plt.xscale('log')
plt.savefig('fig_scatter_improvements_vs_max_distance.png')
plt.show()

plt.scatter(df['improvements'],df['avg_distance'])
plt.xscale('log')
plt.savefig('fig_scatter_improvements_vs_avg_distance.png')
plt.show()

plt.scatter(df['improvements'],df['sum_distance'])
plt.xscale('log')
plt.savefig('fig_scatter_improvements_vs_sum_distance.png')
plt.show()


plt.scatter(df['improvements'],df['min_similarity'])
plt.xscale('log')
plt.savefig('fig_scatter_improvements_vs_min_similarity.png')
plt.show()

plt.scatter(df['improvements'],df['max_similarity'])
plt.xscale('log')
plt.savefig('fig_scatter_improvements_vs_max_similarity.png')
plt.show()

plt.scatter(df['improvements'],df['avg_similarity'])
plt.xscale('log')
plt.savefig('fig_scatter_improvements_vs_avg_similarity.png')
plt.show()

plt.scatter(df['improvements'],df['sum_similarity'])
plt.xscale('log')
plt.savefig('fig_scatter_improvements_vs_sum_similarity.png')
plt.show()
