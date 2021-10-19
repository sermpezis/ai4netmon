#!/usr/bin/env python3
import mongo_util as mng
import bgpdumps_util as bgp
import sys
import time 
from pymongo import DESCENDING, ASCENDING


# import gzip
# import json
# import numpy as np
# from matplotlib import pyplot as plt
# from statsmodels.distributions.empirical_distribution import ECDF
# from collections import defaultdict
# import os.path
# from scipy.sparse import csr_matrix
# from scipy.sparse import dok_matrix



MONGO_DB_NAME = 'asn_distances_v4'
INPUT_STREAM = sys.stdin
DEFAULT_ROUTES = ['0.0.0.0/0', '::/0']


# create/load db
db = mng.get_mongo_db(MONGO_DB_NAME)
t0 = time.time()
if db is None:
   print('Creating new db...')
   db = mng.create_new_db(MONGO_DB_NAME)

   # create collections
   asn2asn_col = db['asn2asn']
   asn2asn_col.create_index([('oa',ASCENDING),('ma',ASCENDING)])

   ij = 0

   for msg in bgp.load_data_ripe_bgp_dumps(INPUT_STREAM):
      ij+=1
      # print('{} \r'.format(ij), end =" ")
      (peer_ip, peer_asn, pfx, path) = msg
      if pfx in DEFAULT_ROUTES:
         continue
      if ':' in pfx:
         continue
      path = bgp.clean_path(path)
      pathlen = len(path)
      origin_asn = path[-1]
      for i, asn in enumerate(path):
         query = { 'oa': int(origin_asn), 'ma': int(asn)}
         value = {'$min': {'pathlen': pathlen-i}}
         asn2asn_col.update_one(query, value, upsert=True)
      if ij==10000:
         break
print(time.time()-t0)