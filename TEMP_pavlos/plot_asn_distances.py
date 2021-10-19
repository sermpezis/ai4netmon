#!/usr/bin/env python3
import gzip
import json
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from collections import defaultdict
import os.path
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix



ASN2ASN_DIST_FNAME = './data/asn2asn__only_peers_pfx.json'
FIG_IMSHOW_FNAME = './figures/fig_imshow_asn_visibility_min_dist_IPv4.png'
FIG_ECDF_FNAME = './figures/fig_ecdf_asn_visibility_IPv4.png'



with open(ASN2ASN_DIST_FNAME, 'r') as f:
   asn2asn = json.load(f)




asn_visibility_v4 = []
asn_mindist_v4 = []
asn_maxdist_v4 = []
visibility_mindist_matrix_v4 = np.zeros((11,14))
for o_asn, dict_o_asn in asn2asn.items():
   current_visibility = len(dict_o_asn)
   minlen = min(dict_o_asn.values())
   maxlen = max(dict_o_asn.values())
   asn_visibility_v4.append(current_visibility)
   asn_mindist_v4.append(minlen)
   asn_maxdist_v4.append(maxlen)
   visibility_mindist_matrix_v4[int(current_visibility/50)+1, minlen] +=1




print('PLOTTING...')
#### PLOTS ####
fontsize = 10
fontsize_small = 7
fontsize_large = 15
linewidth = 2

fig = plt.figure(tight_layout=True)
spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=[6,1], height_ratios=[1,6])

ax = fig.add_subplot(spec[1, 0])
plt.imshow(np.log(visibility_mindist_matrix_v4), cmap='YlOrRd', origin='lower', interpolation=None, aspect='auto')
for i in range(11):
    for j in range(14):
      v = visibility_mindist_matrix_v4[i, j]/1000
      if v > 0:
         if v < 1:
            text = str(int(v*1000))
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
plt.title('#ASNs per visibility and min distance', fontsize=fontsize_large)



ax = fig.add_subplot(spec[0, 0])
ax.bar(np.arange(visibility_mindist_matrix_v4.shape[1]),np.sum(visibility_mindist_matrix_v4,axis=0), color='k')
plt.axis([0.5, 13.5, 0, 50000])
plt.xticks(ticks = [])
plt.yticks(ticks = [])
plt.title('Distribution of min distance', fontsize=fontsize)



ax = fig.add_subplot(spec[1, 1])
ax.barh(np.arange(visibility_mindist_matrix_v4.shape[0]),np.sum(visibility_mindist_matrix_v4,axis=1), color='k')
plt.axis([0, 100000, 0.5, 10.5])
plt.xticks(ticks = [])
plt.yticks(ticks = [])
plt.ylabel('Distribution of visibility', fontsize=fontsize)
ax.yaxis.set_label_position("right")


plt.savefig(FIG_IMSHOW_FNAME)
plt.close()



ecdf_v4 = ECDF(asn_visibility_v4)
plt.plot(ecdf_v4.x, ecdf_v4.y, '-k', linewidth=linewidth)
plt.xlabel('Visibility (by #peers)', fontsize=fontsize_large)
plt.ylabel('CDF', fontsize=fontsize_large)
plt.title('Distribution of ASN visibility (by #peers)', fontsize=fontsize_large)
plt.grid()
plt.savefig(FIG_ECDF_FNAME)
plt.close()
