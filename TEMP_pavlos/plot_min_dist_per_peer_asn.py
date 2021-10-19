#!/usr/bin/env python3
import json
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from collections import defaultdict, Counter
import os.path



ASN2ASN_DIST_FNAME = './data/asn2asn__only_peers_pfx.json'
DIST_PER_PEER_FNAME = './data/dist_per_peer_pfx.json'
FIG_ECDF_NB_FEEDS_PER_PEER_FNAME = './figures/fig_ecdf_nb_feeds_per_peer.png'
FIG_SCATTER_NB_FEEDS_VS_AVG_DIST_FNAME = './figures/fig_scatter_nb_feeds_vs_avg_dist.png'
FIG_ECDF_DIST_ALL_PEERS_FNAME = './figures/fig_ecdf_distances_all_peers.png'
FIG_SCATTER_NB_FEEDS_VS_NB_MIN_DIST_PEER_FNAME = './figures/fig_scatter_nb_feeds_vs_nb_min_dist_peer.png'
FIG_SCATTER_NB_FEEDS_VS_PERC_MIN_DIST_PEER_FNAME = './figures/fig_scatter_nb_feeds_vs_perc_min_dist_peer.png'
FIG_SCATTER_NB_FEEDS_VS_NB_UNIQUE_MIN_DIST_PEER_FNAME = './figures/fig_scatter_nb_feeds_vs_nb_unique_min_dist_peer.png'
FIG_SCATTER_NB_FEEDS_VS_PERC_UNIQUE_MIN_DIST_PEER_FNAME = './figures/fig_scatter_nb_feeds_vs_perc_unique_min_dist_peer.png'



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


# # ECDF of distances for each peer 
# for p,d in peer_distances.items():
#    ecdf = ECDF(d)
#    plt.plot(ecdf.x, ecdf.y, '-k', linewidth=0.5)
# plt.xlabel('distance from origin ASNs', fontsize=fontsize_large)
# plt.ylabel('CDF', fontsize=fontsize_large)
# plt.title('Distribution of distances per peer', fontsize=fontsize_large)
# plt.grid()
# plt.savefig(FIG_ECDF_DIST_ALL_PEERS_FNAME)
# plt.close()



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
