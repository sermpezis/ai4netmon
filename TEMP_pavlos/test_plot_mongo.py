#!/usr/bin/env python3
import gzip
import json
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from collections import defaultdict
import os.path
import pymongo

STATS_FNAME = '../../ris-distance/stats.2021.06.01.txt.gz'
DIST_FNAME = './peer.pfx.pathlen_full.2021.07.01.txt.gz'
PEER2PFX_MINDIST_FILENAME = './peer2pfx_mindist.json'




myclient = pymongo.MongoClient("mongodb://localhost:27017/")
db = myclient["pfx_distances"]
pfx2peer_col = db['pfx2peer']
pfx2asn_col = db['pfx2asn']
list_of_pfxs = pfx2peer_col.distinct('pfx')
nb_pfxs = len(list_of_pfxs)

pfx_visibility_v4 = []
pfx_visibility_v6 = []
pfx_mindist_v4 = []
pfx_mindist_v6 = []
pfx_maxdist_v4 = []
pfx_maxdist_v6 = []
i = 0
for pfx in list_of_pfxs:
   i+=1
   print('{}\% \r'.format(round(100.0*i/nb_pfxs,3)), end =" ")
   list_of_pathlens = [x['pathlen'] for x in pfx2peer_col.find({'pfx':pfx}).sort('pathlen',1)]
   if ':' in pfx:
      pfx_visibility_v6.append(len(list_of_pathlens))
      pfx_mindist_v6.append(list_of_pathlens[0])
      pfx_maxdist_v6.append(list_of_pathlens[-1])
   else:
      pfx_visibility_v4.append(len(list_of_pathlens))
      pfx_mindist_v4.append(list_of_pathlens[0])
      pfx_maxdist_v4.append(list_of_pathlens[-1])


print('PLOTTING...')
#### PLOTS ####
fontsize = 10
fontsize_small = 7
fontsize_large = 15

fig = plt.figure(tight_layout=True)
spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=[6,1], height_ratios=[1,6])

ax = fig.add_subplot(spec[1, 0])
plt.imshow(np.log(visibility_mindist_matrix_v4), cmap='YlOrRd', origin='lower', interpolation=None, aspect='auto')
for i in range(11):
    for j in range(14):
      v = visibility_mindist_matrix_v4[i, j]/1000
      if v > 0:
         if v < 1:
            text = '< 1k'
            ax.text(j, i, text, ha="center", va="center", color="k", fontsize=fontsize_small)
         else:
            text = str(int(v))+'k'
            text = text.rjust(5)
            ax.text(j, i, text, ha="center", va="center", color="k", weight="bold", fontsize=fontsize_small)

plt.axis([0.5, 13.5, 0.5, 10.5])
plt.xticks(ticks = list(range(1,14)), fontsize=fontsize)
plt.yticks(ticks = list(range(1,11)), labels = [i*50 for i in range(1,11)], fontsize=fontsize)
plt.xlabel('Min distance', fontsize=fontsize_large)
plt.ylabel('Visibility (by #peers)', fontsize=fontsize_large)
plt.title('#prefixes per visibility and min distance', fontsize=fontsize_large)



ax = fig.add_subplot(spec[0, 0])
ax.bar(np.arange(visibility_mindist_matrix_v4.shape[1]),np.sum(visibility_mindist_matrix_v4,axis=0), color='k')
plt.axis([0.5, 13.5, 0, 500000])
plt.xticks(ticks = [])
plt.yticks(ticks = [])
plt.title('Distribution of min distance', fontsize=fontsize)



ax = fig.add_subplot(spec[1, 1])
ax.barh(np.arange(visibility_mindist_matrix_v4.shape[0]),np.sum(visibility_mindist_matrix_v4,axis=1), color='k')
plt.axis([0, 500000, 0.5, 10.5])
plt.xticks(ticks = [])
plt.yticks(ticks = [])
plt.ylabel('Distribution of visibility', fontsize=fontsize)
ax.yaxis.set_label_position("right")





plt.savefig('ttt111.png')
plt.close()