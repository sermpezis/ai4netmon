#!/usr/bin/env python3
import sys
import gzip
from itertools import groupby 
import socket
import re
import json
from collections import defaultdict

dfd=open('debug','wt')
debug_asns = set(['3549','2914'])

dist_fname = sys.argv[1]
stats_fname = sys.argv[2]
min_power = int( sys.argv[3] ) ##  minimum peers that saw a prefix (some indication that a prefix is of global significance)
TIME_ISO = sys.argv[4]

pfx_mindist = {4: {}, 6: {}}

peer_pwr = {4: {}, 6: {}}

pfx2cc = {}

# DEBUG

## load stats so we only consider pfxes that were seen by minimal peers
with gzip.open(stats_fname,'rt') as inf:
   for line in inf:
      line = line.rstrip('\n')
      fields = line.split()
      if fields[0] == 'PFX':
         # (typ,pfx,pwr,minlen,maxlen,cc) = fields
         (typ,pfx,pwr,minlen,maxlen) = fields
         pwr = int( pwr )
         minlen = int( minlen )
         maxlen = int( maxlen )
         af = 4
         if ':' in pfx:
            af = 6
         if pwr > min_power:
            # pfx2cc[ pfx ] = cc
            pfx_mindist[ af ][ pfx ] = minlen


##
candidates = {} # candidate asns for getting closer to RIS

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
   ### TODO artifact removal: prepending / inpending / route poisoning
   ## for now we simplify to nr of ASNs in the path
   af = 4
   if ':' in pfx:
      af = 6
   if pfx in pfx_mindist[ af ]:
      mindist = pfx_mindist[ af ][ pfx ]
      asns = fields[6].split()
      
      # remove duplicates
      asns_nodup = [i[0] for i in groupby(asns)]
      #if len( asns ) > len( asns_nodup ):
      #   print("PEND", asns, asns_nodup)

      # remove path poisoning
      if asns_nodup.count( asns_nodup[ -1 ] ) > 1:
         # start_i = None
         # # find the fist occurance of 'last asn' at index start_i
         # for idx, asn in enumerate( asns_nodup ):
         #        if asn == asns_nodup[ -1 ]:
         #           start_i = idx
         #           break
         # healthy_path = [] 
         # poison_path = []
         # for idx,asn in enumerate( asns_nodup ):
         #    if idx <= start_i:
         #       healthy_path.append( asn )
         #    else:
         #       poison_path.append( asn )
         start_i = asns_nodup.index(asns_nodup[-1])
         healthy_path = asns_nodup[0:start_i+1]
         poison_path  = asns_nodup[start_i+1:]



         asns_nodup = healthy_path # put it back into the asns_nodup
      # create a reverse version
      asns_r = list( reversed( asns_nodup ) )
      for idx,asn in enumerate( asns_r ):
         dist = idx+1 # asn at idx=0 is at distance 1 from RIS
         if dist < mindist: # it's closer then what we have
            # if asn in debug_asns:
            #    print("%s < %s pfx:%s asn:%s path:%s peer_ip:%s peer_asn:%s rev:%s" % (dist, mindist, pfx, asn, asns_nodup, peer, peer_asn, asns_r), file=dfd)
            key = (pfx,asn)
            if key in candidates: # we saw this ASN/pfx already at a certain distance
               if candidates[ key ] > dist:
                  candidates[ key ] = dist
            else:
                  candidates[ key ] = dist


# asn_score = {4:{}, 6:{}}
# for pfx,asn in candidates.keys():
#    af = 4
#    if ':' in pfx:
#       af = 6
#    dist = candidates[ (pfx,asn) ]
#    improve = pfx_mindist[ af ][ pfx ] - dist # this is the improvement!
#    #DEBUG LOG print "# %s %s %s %s" % ( pfx, asn, d[pfx], val )
#    asn_score[ af ].setdefault( asn, 0 )
#    asn_score[ af ][ asn ] += improve
# 
# for af in (4,6):
#     for asn in sorted( asn_score[ af ].keys(), key=lambda x: asn_score[af][x] ):
#         print( "%s %s %s" % ( af, asn, asn_score[af][ asn ] ) )

ddd = dist_fname.split('.')
xxx = ".".join([ddd[3], ddd[4], ddd[5]])
candidates_to_dump = defaultdict(dict)
for k,v in candidates.items():
   candidates_to_dump[k[0]][k[1]] = v
with open('pfx.asn.min_distance'+xxx+'.json','w') as fp:
   json.dump(candidates_to_dump, fp)


# ### now do the per country scoring
# cc_asn_score = {4:{}, 6:{}}
# cc_set = set()
# for pfx,asn in candidates.keys():
#    af = 4
#    if ':' in pfx:
#       af = 6
#    dist = candidates[ (pfx,asn) ]
#    improve = pfx_mindist[ af ][ pfx ] - dist # this is the improvement!
#    # find the country
#    cc = '--'
#    if pfx in pfx2cc:
#       cc = pfx2cc[ pfx ]
#       cc_set.add( cc )
#       cc_asn_score[ af ].setdefault( cc, {} )
#       cc_asn_score[ af ][ cc ].setdefault( asn, 0 )
#       cc_asn_score[ af ][ cc ][ asn ] += improve

# for cc in sorted( list( cc_set ) ):
#    for af in (4,6):
#       if cc in cc_asn_score[ af ]:
#          for asn in sorted( cc_asn_score[ af ][ cc ].keys(), key=lambda x: cc_asn_score[af][cc][x] ):
#             print( "%s %s %s %s" % ( cc, af, asn, cc_asn_score[af][cc][ asn ] ) )

