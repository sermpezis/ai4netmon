#!/usr/bin/env python3
import sys
import json
import math
from itertools import groupby

d = {} # holds pfxes and their minimum distance


for line in sys.stdin:
    line = line.rstrip('\n')
    fields = line.split('|')
    peer = fields[3]
    peer_asn = fields[4]
    pfx  = fields[5]
    # let's not consider default routes being sent
    if pfx == '0.0.0.0/0':
        continue
    if pfx == '::/0':
        continue
    asns = fields[6].split(' ')
    # remove duplicates
    asns_nodup = [i[0] for i in groupby(asns)]
    has_dups = 0
    has_poison = 0
    if len( asns ) > len( asns_nodup ):
      has_dups = 1
    # remove path poisoning
    if asns_nodup.count( asns_nodup[ -1 ] ) > 1:
         start_i = None
         # find the fist occurance of 'last asn' at index start_i
         for idx, asn in enumerate( asns_nodup ):
                if asn == asns_nodup[ -1 ]:
                   start_i = idx
                   break
         healthy_path = []
         poison_path = []
         for idx,asn in enumerate( asns_nodup ):
            if idx <= start_i:
               healthy_path.append( asn )
            else:
               poison_path.append( asn )
         if len( asns_nodup ) > len( healthy_path ):
            has_poison = 1
         asns_nodup = healthy_path # put it back into the asns_nodup
    nodup_len = len( asns_nodup )
    orig_len = len( set( asns ) )
    print("%s %s %s %s %s %s %s %s" % ( peer, peer_asn, pfx, asns_nodup[ -1 ], nodup_len, orig_len, has_dups, has_poison ))
