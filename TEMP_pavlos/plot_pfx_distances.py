#!/usr/bin/env python3
import gzip
import json
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from collections import defaultdict


STATS_FNAME = '../../ris-distance/stats.2021.06.01.txt.gz'
FIG_IMSHOW_FNAME = './figures/fig_imshow_pfx_visibility_min_dist_IPv{}.png'
FIG_ECDF_FNAME = './figures/fig_ecdf_pfx_visibility.png'


print('LOADING STATS DATA...')
pfx_visibility = defaultdict(list)
pfx_mindist = defaultdict(list)
pfx_maxdist = defaultdict(list)
visibility_mindist_matrix = dict()
visibility_mindist_matrix['4'] = np.zeros((11,14))
visibility_mindist_matrix['6'] = np.zeros((11,14))
with gzip.open(STATS_FNAME,'rt') as inf:
   for line in inf:
      line = line.rstrip('\n')
      fields = line.split()
      if fields[0] == 'PFX':
         (typ,pfx,pwr,minlen,maxlen,cc) = fields
         pwr = int( pwr )
         if ':' in pfx:
            IPv = '6'
         else:
            IPv = '4'
         pfx_visibility[IPv].append(int(pwr))
         pfx_mindist[IPv].append(int(minlen))
         pfx_maxdist[IPv].append(int(maxlen))
         visibility_mindist_matrix[IPv][int(int(pwr)/50)+1, int(minlen)] +=1




print('PLOTTING...')
#### PLOTS ####
fontsize = 10
fontsize_small = 7
fontsize_large = 15
linewidth = 2


for IPv in ['4','6']:
   fig = plt.figure(tight_layout=True)
   spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=[6,1], height_ratios=[1,6])

   ax = fig.add_subplot(spec[1, 0])
   plt.imshow(np.log(visibility_mindist_matrix[IPv]), cmap='YlOrRd', origin='lower', interpolation=None, aspect='auto')
   for i in range(11):
       for j in range(14):
         v = visibility_mindist_matrix[IPv][i, j]/1000
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
   ax.bar(np.arange(visibility_mindist_matrix[IPv].shape[1]),np.sum(visibility_mindist_matrix[IPv],axis=0), color='k')
   plt.axis([0.5, 13.5, 0, 500000])
   plt.xticks(ticks = [])
   plt.yticks(ticks = [])
   plt.title('Distribution of min distance', fontsize=fontsize)

   ax = fig.add_subplot(spec[1, 1])
   ax.barh(np.arange(visibility_mindist_matrix[IPv].shape[0]),np.sum(visibility_mindist_matrix[IPv],axis=1), color='k')
   plt.axis([0, 500000, 0.5, 10.5])
   plt.xticks(ticks = [])
   plt.yticks(ticks = [])
   plt.ylabel('Distribution of visibility', fontsize=fontsize)
   ax.yaxis.set_label_position("right")

   plt.savefig(FIG_IMSHOW_FNAME.format(IPv))
   plt.close()



ecdf_v4 = ECDF(pfx_visibility['4'])
ecdf_v6 = ECDF(pfx_visibility['6'])
plt.plot(ecdf_v4.x, ecdf_v4.y, '-k', ecdf_v6.x, ecdf_v6.y, '--k', linewidth=linewidth)
plt.xlabel('Visibility (by #peers)', fontsize=fontsize_large)
plt.ylabel('CDF', fontsize=fontsize_large)
plt.title('Distribution of visibility (by #peers)', fontsize=fontsize_large)
plt.legend(['IPv4', 'IPv6'], fontsize=fontsize_large, loc='upper left')
plt.grid()
plt.savefig(FIG_ECDF_FNAME)
plt.close()