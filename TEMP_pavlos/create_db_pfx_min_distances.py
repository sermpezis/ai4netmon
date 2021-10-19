#!/usr/bin/env python3
import mongo_util as mng
import bgpdumps_util as bgp
import sys
# import time 
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



MONGO_DB_NAME = 'pfx_distances'
INPUT_STREAM = sys.stdin
DEFAULT_ROUTES = ['0.0.0.0/0', '::/0']


# create/load db
db = mng.get_mongo_db(MONGO_DB_NAME)

if db is None:
   print('Creating new db...')
   db = mng.create_new_db(MONGO_DB_NAME)

   # create collections
   pfx2peer_col = db['pfx2peer']
   pfx2asn_col = db['pfx2asn']
   pfx2asn_col.create_index([('pfx',ASCENDING),('peer_asn',ASCENDING)])

   for msg in bgp.load_data_ripe_bgp_dumps(INPUT_STREAM):
      (peer_ip, peer_asn, pfx, path) = msg
      if pfx in DEFAULT_ROUTES:
         continue
      path = bgp.clean_path(path)
      pathlen = len(path)
      d = { 'pfx': pfx, 'peer_ip':peer_ip, 'peer_asn': int(peer_asn) , 'pathlen':pathlen}
      pfx2peer_col.insert_one(d)
      for i, asn in enumerate(path):
         query = { 'pfx': pfx, 'peer_asn': asn}
         value = {'$min': {'pathlen': pathlen-i}}
         pfx2asn_col.update_one(query, value, upsert=True)