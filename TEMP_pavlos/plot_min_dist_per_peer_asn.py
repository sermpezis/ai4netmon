#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from collections import defaultdict, Counter
import os.path
import sys
import Analysis.aggregate_data.data_aggregation_tools as dat



ASN2ASN_DIST_FNAME = './data/asn2asn__only_peers_pfx.json'
DIST_PER_PEER_FNAME = './data/dist_per_peer_pfx.json'
LIST_OF_RIPE_PEERS_FNAME = './list_of_RIPE_RIS_peers.json'

FIG_ECDF_NB_FEEDS_PER_PEER_FNAME = './figures/fig_ecdf_nb_feeds_per_peer.png'
FIG_SCATTER_NB_FEEDS_VS_AVG_DIST_FNAME = './figures/fig_scatter_nb_feeds_vs_avg_dist.png'
FIG_ECDF_DIST_ALL_PEERS_FNAME = './figures/fig_ecdf_distances_all_peers.png'
FIG_ECDF_DIST_ALL_PEERS_PER_TIER_FNAME = './figures/fig_ecdf_distances_all_peers_per_tier.png'
FIG_SCATTER_NB_FEEDS_VS_NB_MIN_DIST_PEER_FNAME = './figures/fig_scatter_nb_feeds_vs_nb_min_dist_peer.png'
FIG_SCATTER_NB_FEEDS_VS_PERC_MIN_DIST_PEER_FNAME = './figures/fig_scatter_nb_feeds_vs_perc_min_dist_peer.png'
FIG_SCATTER_NB_FEEDS_VS_NB_UNIQUE_MIN_DIST_PEER_FNAME = './figures/fig_scatter_nb_feeds_vs_nb_unique_min_dist_peer.png'
FIG_SCATTER_NB_FEEDS_VS_PERC_UNIQUE_MIN_DIST_PEER_FNAME = './figures/fig_scatter_nb_feeds_vs_perc_unique_min_dist_peer.png'
FIG_BAR_CORR_VARS_AVG_DIST_FNAME = './figures/fig_bar_corr_characteristics_vs_avg_dist.png'
FIG_BAR_CORR_VARS_AVG_DIST_ONLY_FULL_FEED_FNAME = './figures/fig_bar_corr_characteristics_vs_avg_dist_only_full_feed.png'
FIG_SCATTER_AVG_DIST_VS_NB_NEIGH_TOTAL_FNAME = './figures/fig_scatter_avg_dist_vs_nb_neigh_total.png'
FIG_SCATTER_AVG_DIST_VS_NB_NEIGH_PEERS_FNAME = './figures/fig_scatter_avg_dist_vs_nb_neigh_peers.png'
FIG_SCATTER_AVG_DIST_VS_NB_IXPS_FNAME = './figures/fig_scatter_avg_dist_vs_nb_ixps.png'
FIG_BAR_CORR_VARS_IMPROVEMENT_FNAME = './figures/fig_bar_corr_characteristics_vs_improvement.png'
FIG_BAR_CORR_VARS_IMPROVEMENT_ONLY_FULL_FEED_FNAME = './figures/fig_bar_corr_characteristics_vs_improvement_only_full_feed.png'
FIG_SCATTER_IMPROVEMENT_VS_NB_NEIGH_CUSTOMERS_FNAME = './figures/fig_scatter_improvement_vs_nb_neigh_customers.png'
FIG_SCATTER_IMPROVEMENT_VS_NB_NEIGH_PEERS_FNAME = './figures/fig_scatter_improvement_vs_nb_neigh_peers.png'
FIG_SCATTER_IMPROVEMENT_VS_CUST_CONE_FNAME = './figures/fig_scatter_improvement_vs_cust_cone.png'
FIG_SCATTER_IMPROVEMENT_VS_NB_IXPS_FNAME = './figures/fig_scatter_improvement_vs_nb_ixps.png' 
FIG_SCATTER_IMPROVEMENT_VS_NB_FAC_FNAME = './figures/fig_scatter_improvement_vs_nb_facilities.png' 

print('Loading asn2asn distance data ...')
with open(ASN2ASN_DIST_FNAME, 'r') as f:
   asn2asn = json.load(f)


if not os.path.isfile(DIST_PER_PEER_FNAME):
   print('Creating lists per peer ...')
   l = len(asn2asn.keys())
   peer_distances = defaultdict(list)
   for o_asn, dict_o_asn in asn2asn.items():
      for peer, dist in dict_o_asn.items():
         peer_distances[peer].append(dist)

   print('Saving lists per peer ...')
   with open(DIST_PER_PEER_FNAME, 'w') as f:
      json.dump(peer_distances,f)
else:
   print('Loading lists per peer from file...')
   with open(DIST_PER_PEER_FNAME, 'r') as f:
      peer_distances = json.load(f)  



print('Calculating analytics ...')
nb_of_feeds_per_peer = {p:len(l) for p,l in peer_distances.items()}

avg_dist_per_peer = {p:np.mean(l) for p,l in peer_distances.items()}

min_distance_peers = []
min_distance_peers_unique = []
for o_asn, dict_o_asn in asn2asn.items():
   min_dist = min(dict_o_asn.values())
   list_of_min_dist_peers = [p for p,d in dict_o_asn.items() if d==min_dist]
   min_distance_peers.extend([p for p,d in dict_o_asn.items() if d==min_dist])
   if len(list_of_min_dist_peers) == 1:
      min_distance_peers_unique.extend(list_of_min_dist_peers)
# print(min_distance_peers)
nb_min_dist_peer = Counter(min_distance_peers) 
nb_min_dist_peer_unique = Counter(min_distance_peers_unique) 

input_clique_AS_rel = '174 209 286 701 1239 1299 2828 2914 3257 3320 3356 3491 5511 6453 6461 6762 6830 7018 12956'
list_of_tier1_asns = [int(i) for i in input_clique_AS_rel.split(' ')]
with open(LIST_OF_RIPE_PEERS_FNAME,'r') as f:
   dict_ripe = json.load(f)
list_of_tier1_peers = [addr for addr,asn in dict_ripe.items() if asn in list_of_tier1_asns]
distances_tier1 = []
distances_non_tier1 = []
for p,l in peer_distances.items():
   if p in list_of_tier1_peers:
      distances_tier1.extend(l)
   else:
      distances_non_tier1.extend(l)


peer_asn_distances = defaultdict(list)
for p,l in peer_distances.items():
   peer_asn_distances[dict_ripe[p]].extend(l)
avg_dist_per_peer_asn = {p:np.mean(l) for p,l in peer_asn_distances.items()}

feeds_per_peer_asn = defaultdict(list)
for o_asn, dict_o_asn in asn2asn.items():
   for p in dict_o_asn.keys():
      feeds_per_peer_asn[dict_ripe[p]].extend([o_asn])
nb_of_feeds_per_peer_asn = {p:len(set(l)) for p,l in feeds_per_peer_asn.items()}

improvements = defaultdict(lambda :0)
for o_asn, dict_o_asn in asn2asn.items():
   sorted_dist = sorted(set(dict_o_asn.values()))
   list_of_min_dist_peers = [dict_ripe[p] for p,d in dict_o_asn.items() if d==sorted_dist[0]]
   if len(list_of_min_dist_peers) == 1:
      if len(sorted_dist) > 1:
         improvements[list_of_min_dist_peers[0]] += sorted_dist[1]-sorted_dist[0]

df = pd.read_csv('dataframe_ripe_peers.csv', index_col='AS_rank_asn')
df = df.join(pd.DataFrame.from_dict(avg_dist_per_peer_asn, orient='index',columns=['avg_dist']))
df = df.join(pd.DataFrame.from_dict(improvements, orient='index',columns=['improvements']))
df = df.fillna(value=np.nan)
corr = df.corr() 
print('Correlation matrix')
print(corr)


df_full_feeds = df.join(pd.DataFrame.from_dict(nb_of_feeds_per_peer_asn, orient='index',columns=['nb_feeds']))
df_full_feeds = df_full_feeds[df_full_feeds['nb_feeds'] > 60000]
corr_full_feed = df_full_feeds.corr()
print('Correlation matrix (only full feeders)')
print(corr_full_feed)





print('PLOTTING...')
#### PLOTS ####
fontsize = 10
fontsize_small = 7
fontsize_large = 15
linewidth = 2
markersize = 10


# ECDF plot: nb of feeds per peer
ecdf = ECDF(list(nb_of_feeds_per_peer.values()))
plt.plot(ecdf.x, ecdf.y, '-k', linewidth=linewidth)
plt.xlabel('#feeds (in #origin_ASNs))', fontsize=fontsize_large)
plt.ylabel('CDF', fontsize=fontsize_large)
plt.title('Distribution of #feeds per peer', fontsize=fontsize_large)
plt.xscale('log')
plt.grid()
plt.savefig(FIG_ECDF_NB_FEEDS_PER_PEER_FNAME)
plt.close()


# scatter plot: nb of feeds vs avg dist
list_of_peers = list(peer_distances.keys())
f = [nb_of_feeds_per_peer[p] for p in list_of_peers]
d = [avg_dist_per_peer[p] for p in list_of_peers]
plt.scatter(f,d, s=markersize)
plt.xlabel('#feeds (in #origin_ASNs))', fontsize=fontsize_large)
plt.ylabel('avg. distance', fontsize=fontsize_large)
plt.title('#feeds vs. avg_distance per peer', fontsize=fontsize_large)
plt.xscale('log')
plt.grid()
plt.savefig(FIG_SCATTER_NB_FEEDS_VS_AVG_DIST_FNAME)
plt.close()


# ECDF of distances for each peer 
for p,d in peer_distances.items():
   ecdf = ECDF(d)
   if p in list_of_tier1_peers:
      plt.plot(ecdf.x, ecdf.y, '-r', linewidth=2)
   else:
      plt.plot(ecdf.x, ecdf.y, '--k', linewidth=0.5)
h = plt.plot(0,0,'-r',0,0,'--k')
plt.legend(h,['tier-1 peers', 'non-tier1 peers'],loc='lower right')
plt.xlabel('distance from origin ASNs', fontsize=fontsize_large)
plt.ylabel('CDF', fontsize=fontsize_large)
plt.title('Distribution of distances per peer', fontsize=fontsize_large)
plt.grid()
plt.savefig(FIG_ECDF_DIST_ALL_PEERS_FNAME)
plt.close()


# ECDF of distances for tier1 peers and non-tier1 peers
ecdf = ECDF(distances_tier1)
plt.plot(ecdf.x, ecdf.y, '-r', linewidth=linewidth)
ecdf = ECDF(distances_non_tier1)
plt.plot(ecdf.x, ecdf.y, '-k', linewidth=linewidth)
plt.legend(['tier-1 peers', 'non-tier1 peers'],loc='lower right')
plt.xlabel('distance from origin ASNs', fontsize=fontsize_large)
plt.ylabel('CDF', fontsize=fontsize_large)
plt.title('Distribution of distances for Tier-1 and non-Tier-1 peers', fontsize=fontsize_large)
plt.grid()
plt.savefig(FIG_ECDF_DIST_ALL_PEERS_PER_TIER_FNAME)
plt.close()



# scatter plot: nb of feeds vs nb occurences as min dist peer
list_of_peers = list(peer_distances.keys())
f = [nb_of_feeds_per_peer[p] for p in list_of_peers]
o = [nb_min_dist_peer[p] for p in list_of_peers]
plt.scatter(f,o, s=markersize)
plt.xlabel('#feeds (in #origin_ASNs))', fontsize=fontsize_large)
plt.ylabel('#occurences as min dist peer', fontsize=fontsize_large)
plt.title('#feeds vs. #occurences as min dist peer', fontsize=fontsize_large)
plt.xscale('log')
plt.grid()
plt.savefig(FIG_SCATTER_NB_FEEDS_VS_NB_MIN_DIST_PEER_FNAME)
plt.close()


# scatter plot: nb of feeds vs percentage occurences as min dist peer
list_of_peers = list(peer_distances.keys())
f = [nb_of_feeds_per_peer[p] for p in list_of_peers]
o = [nb_min_dist_peer[p]/nb_of_feeds_per_peer[p] for p in list_of_peers]
plt.scatter(f,o, s=markersize)
plt.xlabel('#feeds (in #origin_ASNs))', fontsize=fontsize_large)
plt.ylabel('% occurences as min dist peer', fontsize=fontsize_large)
plt.title('#feeds vs. % occurences as min dist peer', fontsize=fontsize_large)
plt.xscale('log')
plt.grid()
plt.savefig(FIG_SCATTER_NB_FEEDS_VS_PERC_MIN_DIST_PEER_FNAME)
plt.close()

# scatter plot: nb of feeds vs nb occurences as unique min dist peer
list_of_peers = list(peer_distances.keys())
f = [nb_of_feeds_per_peer[p] for p in list_of_peers]
o = [nb_min_dist_peer_unique[p] for p in list_of_peers]
plt.scatter(f,o, s=markersize)
plt.xlabel('#feeds (in #origin_ASNs))', fontsize=fontsize_large)
plt.ylabel('#occurences as unique min dist peer', fontsize=fontsize_large)
plt.title('#feeds vs. #occurences as unique min dist peer', fontsize=fontsize_large)
plt.xscale('log')
plt.grid()
plt.savefig(FIG_SCATTER_NB_FEEDS_VS_NB_UNIQUE_MIN_DIST_PEER_FNAME)
plt.close()


# scatter plot: nb of feeds vs percentage occurences as unique min dist peer
list_of_peers = list(peer_distances.keys())
f = [nb_of_feeds_per_peer[p] for p in list_of_peers]
o = [nb_min_dist_peer_unique[p]/nb_of_feeds_per_peer[p] for p in list_of_peers]
plt.scatter(f,o, s=markersize)
plt.xlabel('#feeds (in #origin_ASNs))', fontsize=fontsize_large)
plt.ylabel('%occurences as unique min dist peer', fontsize=fontsize_large)
plt.title('#feeds vs. %occurences as unique min dist peer', fontsize=fontsize_large)
plt.xscale('log')
plt.grid()
plt.savefig(FIG_SCATTER_NB_FEEDS_VS_PERC_UNIQUE_MIN_DIST_PEER_FNAME)
plt.close()


# Bar plot: correlations between variables and avg distance
corr_variables = ['AS_rank_numberAsns', 'AS_rank_total', 'AS_rank_customer', 'AS_rank_peer', 'AS_rank_provider', 'peeringDB_ix_count', 'peeringDB_fac_count']
corr_var_names = ['Customer cone (#ASNs)', '#neighbors (total)', '#neighbors (customers)', '#neighbors (peers)', '#neighbors (providers)', '#IXPs (PDB)', '#facilities (PDB)']
corr.loc[corr_variables,['avg_dist']].plot.barh(legend=False)
plt.yticks(range(len(corr_variables)),labels=corr_var_names)
plt.subplots_adjust(left=0.3)
plt.xlabel('pearson correlation coef.', fontsize=fontsize)
plt.title('network characteristics vs avg distance per peer', fontsize=fontsize_large)
plt.grid()
plt.savefig(FIG_BAR_CORR_VARS_AVG_DIST_FNAME)
plt.close()

# Bar plot: correlations between variables and avg distance (only full feed)
corr_variables = ['AS_rank_numberAsns', 'AS_rank_total', 'AS_rank_customer', 'AS_rank_peer', 'AS_rank_provider', 'peeringDB_ix_count', 'peeringDB_fac_count']
corr_var_names = ['Customer cone (#ASNs)', '#neighbors (total)', '#neighbors (customers)', '#neighbors (peers)', '#neighbors (providers)', '#IXPs (PDB)', '#facilities (PDB)']
corr_full_feed.loc[corr_variables,['avg_dist']].plot.barh(legend=False)
plt.yticks(range(len(corr_variables)),labels=corr_var_names)
plt.subplots_adjust(left=0.3)
plt.xlabel('pearson correlation coef.', fontsize=fontsize)
plt.title('network characteristics vs avg distance per peer', fontsize=fontsize_large)
plt.grid()
plt.savefig(FIG_BAR_CORR_VARS_AVG_DIST_ONLY_FULL_FEED_FNAME)
plt.close()


# scatter plot: avg distance vs #neighbors (total)
df.plot.scatter(x = 'avg_dist', y = 'AS_rank_total', s=markersize)
plt.xlabel('average distance', fontsize=fontsize_large)
plt.ylabel('#neighbors (total)', fontsize=fontsize_large)
plt.title('avg distance vs #neighbors (total)', fontsize=fontsize_large)
plt.yscale('log')
plt.axis([0,6,1,10000])
plt.grid()
plt.savefig(FIG_SCATTER_AVG_DIST_VS_NB_NEIGH_TOTAL_FNAME)
plt.close()

# scatter plot: avg distance vs #neighbors (peers)
df.plot.scatter(x = 'avg_dist', y = 'AS_rank_peer', s=markersize)
plt.xlabel('average distance', fontsize=fontsize_large)
plt.ylabel('#neighbors (peers)', fontsize=fontsize_large)
plt.title('avg distance vs #neighbors (peers)', fontsize=fontsize_large)
plt.yscale('log')
plt.axis([0,6,1,10000])
plt.grid()
plt.savefig(FIG_SCATTER_AVG_DIST_VS_NB_NEIGH_PEERS_FNAME)
plt.close()

# scatter plot: avg distance vs #IXPs
df.plot.scatter(x = 'avg_dist', y = 'peeringDB_ix_count', s=markersize)
plt.xlabel('average distance', fontsize=fontsize_large)
plt.ylabel('#IXPs (PDB)', fontsize=fontsize_large)
plt.title('avg distance vs #IXPs (PDB)', fontsize=fontsize_large)
plt.yscale('log')
plt.axis([0,6,1,1000])
plt.grid()
plt.savefig(FIG_SCATTER_AVG_DIST_VS_NB_IXPS_FNAME)
plt.close()




# Bar plot: correlations between variables and improvement
corr_variables = ['AS_rank_numberAsns', 'AS_rank_total', 'AS_rank_customer', 'AS_rank_peer', 'AS_rank_provider', 'peeringDB_ix_count', 'peeringDB_fac_count']
corr_var_names = ['Customer cone (#ASNs)', '#neighbors (total)', '#neighbors (customers)', '#neighbors (peers)', '#neighbors (providers)', '#IXPs (PDB)', '#facilities (PDB)']
corr.loc[corr_variables,['improvements']].plot.barh(legend=False)
plt.yticks(range(len(corr_variables)),labels=corr_var_names)
plt.subplots_adjust(left=0.3)
plt.xlabel('pearson correlation coef.', fontsize=fontsize)
plt.title('network characteristics vs improvement per peer', fontsize=fontsize_large)
plt.grid()
plt.savefig(FIG_BAR_CORR_VARS_IMPROVEMENT_FNAME)
plt.close()


# Bar plot: correlations between variables and improvement(only full feeders)
corr_variables = ['AS_rank_numberAsns', 'AS_rank_total', 'AS_rank_customer', 'AS_rank_peer', 'AS_rank_provider', 'peeringDB_ix_count', 'peeringDB_fac_count']
corr_var_names = ['Customer cone (#ASNs)', '#neighbors (total)', '#neighbors (customers)', '#neighbors (peers)', '#neighbors (providers)', '#IXPs (PDB)', '#facilities (PDB)']
corr_full_feed.loc[corr_variables,['improvements']].plot.barh(legend=False)
plt.yticks(range(len(corr_variables)),labels=corr_var_names)
plt.subplots_adjust(left=0.3)
plt.xlabel('pearson correlation coef.', fontsize=fontsize)
plt.title('network characteristics vs improvement per peer', fontsize=fontsize_large)
plt.grid()
plt.savefig(FIG_BAR_CORR_VARS_IMPROVEMENT_ONLY_FULL_FEED_FNAME)
plt.close()

# scatter plot: improvements vs #neighbors (customer)
df.plot.scatter(x = 'improvements', y = 'AS_rank_customer', s=markersize)
plt.xlabel('improvements', fontsize=fontsize_large)
plt.ylabel('#neighbors (customers)', fontsize=fontsize_large)
plt.title('improvement vs #neighbors (customers)', fontsize=fontsize_large)
plt.xscale('log')
plt.yscale('log')
plt.axis([1,5000,1,10000])
plt.grid()
plt.savefig(FIG_SCATTER_IMPROVEMENT_VS_NB_NEIGH_CUSTOMERS_FNAME)
plt.close()

# scatter plot: improvements vs #neighbors (peers)
df.plot.scatter(x = 'improvements', y = 'AS_rank_peer', s=markersize)
plt.xlabel('improvements', fontsize=fontsize_large)
plt.ylabel('#neighbors (peers)', fontsize=fontsize_large)
plt.title('improvement vs #neighbors (peers)', fontsize=fontsize_large)
plt.xscale('log')
plt.yscale('log')
plt.axis([1,5000,1,10000])
plt.grid()
plt.savefig(FIG_SCATTER_IMPROVEMENT_VS_NB_NEIGH_PEERS_FNAME)
plt.close()

# scatter plot: improvements vs customer core
df.plot.scatter(x = 'improvements', y = 'AS_rank_numberAsns', s=markersize)
plt.xlabel('improvements', fontsize=fontsize_large)
plt.ylabel('customer cone', fontsize=fontsize_large)
plt.title('improvement vs customer cone', fontsize=fontsize_large)
plt.xscale('log')
plt.yscale('log')
plt.axis([1,5000,1,10000])
plt.grid()
plt.savefig(FIG_SCATTER_IMPROVEMENT_VS_CUST_CONE_FNAME)
plt.close()

# scatter plot: improvements vs customer cone
df.plot.scatter(x = 'improvements', y = 'AS_rank_numberAsns', s=markersize)
plt.xlabel('improvements', fontsize=fontsize_large)
plt.ylabel('customer cone', fontsize=fontsize_large)
plt.title('improvement vs customer cone', fontsize=fontsize_large)
plt.xscale('log')
plt.yscale('log')
plt.axis([1,5000,1,50000])
plt.grid()
plt.savefig(FIG_SCATTER_IMPROVEMENT_VS_CUST_CONE_FNAME)
plt.close()

# scatter plot: improvements vs #IXPs
df.plot.scatter(x = 'improvements', y = 'peeringDB_ix_count', s=markersize)
plt.xlabel('improvements', fontsize=fontsize_large)
plt.ylabel('#IXPs (PDB)', fontsize=fontsize_large)
plt.title('improvements vs #IXPs (PDB)', fontsize=fontsize_large)
plt.yscale('log')
plt.xscale('log')
plt.axis([1,5000,1,1000])
plt.grid()
plt.savefig(FIG_SCATTER_IMPROVEMENT_VS_NB_IXPS_FNAME)
plt.close()

# scatter plot: improvements vs #facilities
df.plot.scatter(x = 'improvements', y = 'peeringDB_fac_count', s=markersize)
plt.xlabel('improvements', fontsize=fontsize_large)
plt.ylabel('#facilities (PDB)', fontsize=fontsize_large)
plt.title('improvements vs #facilities (PDB)', fontsize=fontsize_large)
plt.yscale('log')
plt.xscale('log')
plt.axis([1,5000,1,1000])
plt.grid()
plt.savefig(FIG_SCATTER_IMPROVEMENT_VS_NB_FAC_FNAME)
plt.close()