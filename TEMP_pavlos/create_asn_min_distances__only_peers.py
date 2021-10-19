#!/usr/bin/env python3
import mongo_util as mng
import bgpdumps_util as bgp
import sys
import time 
from pymongo import DESCENDING, ASCENDING
from collections import defaultdict


# import gzip
import json
# import numpy as np
# from matplotlib import pyplot as plt
# from statsmodels.distributions.empirical_distribution import ECDF
# from collections import defaultdict
# import os.path
# from scipy.sparse import csr_matrix
# from scipy.sparse import dok_matrix



INPUT_STREAM = sys.stdin
DEFAULT_ROUTES = ['0.0.0.0/0', '::/0']
LARGE_PATH_LENGTH = 10000
BATCH_SIZE = 1000


# create/load db
t00 = time.time()
t0 = time.time()
test_dict = defaultdict(dict)


ij = 0
for msg in bgp.load_data_ripe_bgp_dumps(INPUT_STREAM):
   ij+=1
   if ij%1000000 ==0:
      print('{} M: {} sec'.format(int(ij/1000000),int(time.time()-t0)))
      t0 = time.time()
   # if ij==10000000:
   #    break
   # print('{} \r'.format(ij), end =" ")
   (peer_ip, peer_asn, pfx, path) = msg
   if pfx in DEFAULT_ROUTES:
      continue
   if ':' in pfx:
      continue
   path = bgp.clean_path(path)
   pathlen = len(path)
   origin_asn = path[-1]
   test_dict[int(origin_asn)][int(peer_asn)] = min(pathlen, test_dict[int(origin_asn)].get(int(peer_asn),LARGE_PATH_LENGTH)) 
   

with open('test_asn2asn__only_peers.json', 'w') as f:
   json.dump(test_dict,f)

print(time.time()-t00)