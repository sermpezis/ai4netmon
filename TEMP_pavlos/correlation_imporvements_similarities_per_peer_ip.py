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
IMPROVEMENTS_PER_PEER_IP_LEAVE_ONE_OUT_FNAME = './data/improvements_RIPE_RIS_peers_per_ip_leave_one_out.json'
SAVE_FIG_FORMAT = './figures/fig_scatter_improvements_vs_{}.png'
FIG_CORR_FNAME = './figures/fig_correlations_improvements_vs_distances.png'



print('Loading RIPE RIS peers dataset')
with open(LIST_OF_RIPE_PEERS_FNAME,'r') as f:
   dict_ripe = json.load(f)

print('Loading distances sum dataset')
with open(PATHLEN_DISTANCES_SUM_FNAME,'r') as f:
    sum_distance = json.load(f)

print('Loading distances avg dataset')
with open(PATHLEN_DISTANCES_AVG_FNAME,'r') as f:
    avg_distance = json.load(f)

print('Loading distances min dataset')
with open(PATHLEN_DISTANCES_MIN_FNAME,'r') as f:
    min_distance = json.load(f)

print('Loading distances max dataset')
with open(PATHLEN_DISTANCES_MAX_FNAME,'r') as f:
    max_distance = json.load(f)

print('Loading distances sum 5max dataset')
with open(PATHLEN_DISTANCES_SUM_5MAX_FNAME,'r') as f:
    sum_5max_distance = json.load(f)

print('Loading distances sum 10max dataset')
with open(PATHLEN_DISTANCES_SUM_10MAX_FNAME,'r') as f:
    sum_10max_distance = json.load(f)

print('Loading distances sum 5min dataset')
with open(PATHLEN_DISTANCES_SUM_5MIN_FNAME,'r') as f:
    sum_5min_distance = json.load(f)

print('Loading distances sum 10min dataset')
with open(PATHLEN_DISTANCES_SUM_10MIN_FNAME,'r') as f:
    sum_10min_distance = json.load(f)



if not os.path.isfile(IMPROVEMENTS_PER_PEER_IP_LEAVE_ONE_OUT_FNAME):
   print('Loading asn2asn dataset')
   with open(ASN2ASN_DIST_FNAME, 'r') as f:
      asn2asn = json.load(f)

   print('Calculating improvements')
   improvements = defaultdict(lambda :0)
   for o_asn, dict_o_asn in asn2asn.items():
      sorted_dist = sorted(set(dict_o_asn.values()))
      list_of_min_dist_peers = [p for p,d in dict_o_asn.items() if d==sorted_dist[0]]
      if len(list_of_min_dist_peers) == 1:
         if len(sorted_dist) > 1:
            improvements[list_of_min_dist_peers[0]] += sorted_dist[1]-sorted_dist[0]

   print('Saving improvements dataset')
   with open(IMPROVEMENTS_PER_PEER_IP_LEAVE_ONE_OUT_FNAME, 'w') as f:
      json.dump(improvements,f)
else:
   print('Loading improvements dataset')
   with open(IMPROVEMENTS_PER_PEER_IP_LEAVE_ONE_OUT_FNAME, 'r') as f:
      improvements = json.load(f)

print('Calculating total distances')
sum_distance_asn = {ip:sum_distance.get(ip,np.nan) for ip in dict_ripe.keys()}
avg_distance_asn = {ip:avg_distance.get(ip,np.nan) for ip in dict_ripe.keys()}
min_distance_asn = {ip:min_distance.get(ip,np.nan) for ip in dict_ripe.keys()}
max_distance_asn = {ip:max_distance.get(ip,np.nan) for ip in dict_ripe.keys()}
sum_5max_distance_asn = {ip:sum_5max_distance.get(ip,np.nan) for ip in dict_ripe.keys()}
sum_10max_distance_asn = {ip:sum_10max_distance.get(ip,np.nan) for ip in dict_ripe.keys()}
sum_5min_distance_asn = {ip:sum_5min_distance.get(ip,np.nan) for ip in dict_ripe.keys()}
sum_10min_distance_asn = {ip:sum_10min_distance.get(ip,np.nan) for ip in dict_ripe.keys()}



df = pd.DataFrame()
df = pd.DataFrame.from_dict(improvements, orient='index',columns=['improvements'])
df = df.join(pd.DataFrame.from_dict(sum_distance_asn, orient='index',columns=['sum_distance']))
df = df.join(pd.DataFrame.from_dict(avg_distance_asn, orient='index',columns=['avg_distance']))
df = df.join(pd.DataFrame.from_dict(min_distance_asn, orient='index',columns=['min_distance']))
df = df.join(pd.DataFrame.from_dict(max_distance_asn, orient='index',columns=['max_distance']))
df = df.join(pd.DataFrame.from_dict(sum_5max_distance_asn, orient='index',columns=['sum_5max_distance']))
df = df.join(pd.DataFrame.from_dict(sum_10max_distance_asn, orient='index',columns=['sum_10max_distance']))
df = df.join(pd.DataFrame.from_dict(sum_5min_distance_asn, orient='index',columns=['sum_5min_distance']))
df = df.join(pd.DataFrame.from_dict(sum_10min_distance_asn, orient='index',columns=['sum_10min_distance']))
# df['sum_similarity'] = 1/df['sum_distance']
# df['avg_similarity'] = 1/df['avg_distance']
# df['min_similarity'] = 1/df['max_distance']
# df['max_similarity'] = 1/df['min_distance']
df.replace(np.inf,np.nan,inplace=True)
print(df)


print('Correlation matrix')
corr = df.corr() 
print(corr)





print('PLOTTING...')
#### PLOTS ####
fontsize = 10
fontsize_small = 7
fontsize_large = 15
linewidth = 2
markersize = 10


# plot scatter plots
for c in df.columns:
   if c == 'improvements':
      continue
   plt.scatter(df['improvements'],df[c])
   plt.xscale('log')
   plt.savefig(SAVE_FIG_FORMAT.format(c))
   plt.close()


# plot correlation matrix
corr_variables = [i for i in df.columns if i != 'improvements']
corr.loc[corr_variables,['improvements']].plot.barh(legend=False)
plt.yticks(range(len(corr_variables)),labels=corr_variables)
plt.subplots_adjust(left=0.3)
plt.xlabel('pearson correlation coef.', fontsize=fontsize)
plt.title('distances vs improvement per peer', fontsize=fontsize_large)
plt.axis([-1,1,-0.5,len(corr_variables)-0.5])
plt.grid()
plt.savefig(FIG_CORR_FNAME)
plt.close()