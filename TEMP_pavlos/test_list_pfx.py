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


STATS_FNAME = '../../ris-distance/stats.2021.06.01.txt.gz'
DIST_FNAME = './peer.pfx.pathlen_full.2021.07.01.txt.gz'
PEER2PFX_MINDIST_FILENAME = './peer2pfx_mindist.json'



print('LOADING DIST DATA...')

set_of_pfxs = set()
set_of_addr = set()
dict_pfx_origin = defaultdict(set)
i = 0
j = 1
list_of_thresholds = [100000000*i for i in range(1,4)]
with gzip.open(DIST_FNAME, 'rt') as inf:
   for line in inf:
      # print('{}\% \t{} \r'.format(round(100.0*i/348061052,3),len(set_of_pfxs)-len(set_of_addr)), end =" ")
      print('{} \r'.format(i), end =" ")
      i+=1
      line = line.rstrip('\n')
      fields = line.split()
      if len(fields)==1:
         continue
      pfx = fields[2]
      if ':' in pfx:
         af = 6
         continue
      origin_asn = fields[3]
      # set_of_pfxs.add(pfx)
      # set_of_addr.add(pfx.split('/')[0])
      dict_pfx_origin[pfx].add(origin_asn)
      # if i in list_of_thresholds:
      #    with open('pfx2asn_dict{}.json'.format(j), 'w') as f:
      #       json.dump(dict_pfx_origin, f)
      #    dict_pfx_origin = defaultdict(list)
      #    j+=1


for k,v in dict_pfx_origin.items():
   dict_pfx_origin[k] = list(v)


# print([len(set_of_pfxs),len(set_of_addr),len(set_of_pfxs)-len(set_of_addr)])


with open('pfx2asn_dict__new_all.json'.format(j), 'w') as f:
   json.dump(dict_pfx_origin, f)

