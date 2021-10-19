#!/usr/bin/env python3
import sys
import json
d = {} # holds pfxes and their minimum distance


for line in sys.stdin:
    line = line.rstrip('\n')
    fields = line.split('|')
    peer = fields[3]
    peer_asn = fields[4]
    pfx  = fields[5]
    d[peer] = int(peer_asn)

with open('list_of_RIPE_RIS_peers.json','w') as f:
    json.dump(d,f)
    