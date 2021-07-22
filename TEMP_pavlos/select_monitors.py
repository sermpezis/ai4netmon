#!/usr/bin/env python3
import json
import gzip
from matplotlib import pyplot as plt

YYYY='2021'
MM='07'
DD='01'

DIST_FILENAME = 'pfx.asn.min_distance{}.{}.{}.json'.format(YYYY,MM,DD)
STATS_FILENAME = 'stats_no_country.{}.{}.{}.txt.gz'.format(YYYY,MM,DD)

THRES = 1
MAX_MONITORS = 1000



# load dictionary of min distances
with open(DIST_FILENAME, 'r') as f:
    candidates = json.load(f)



# load stats so we only consider pfxes that were seen by minimal peers
pfx_mindist = {4: {}, 6: {}}
with gzip.open(STATS_FILENAME,'rt') as inf:
   for line in inf:
      line = line.rstrip('\n')
      fields = line.split()
      if fields[0] == 'PFX':
        (typ,pfx,pwr,minlen,maxlen) = fields
        af = 4
        if ':' in pfx:
            af = 6
        if int(pwr) > THRES:
            pfx_mindist[ af ][ pfx ] = int(minlen)


# remove all entries for potential monitors that are larger than current monitors
for pfx in candidates.keys():
    af = 4
    if ':' in pfx:
        af = 6
    for asn in candidates[pfx].keys():
        if candidates[pfx][asn] >= pfx_mindist[ af ][ pfx ]:
            del candidates[pfx][asn]


# calculate aggregate scores
asn_score = {4:{}, 6:{}}
for pfx in candidates.keys():
    af = 4
    if ':' in pfx:
        af = 6
    for asn in candidates[pfx].keys():
        dist = candidates[pfx][asn]
        improve = pfx_mindist[ af ][ pfx ] - dist # this is the improvement!
        asn_score[ af ].setdefault( asn, 0 )
        asn_score[ af ][ asn ] += improve


# IPv4
max_monitors = MAX_MONITORS
sorted_asns = sorted(asn_score[4].keys(), key=asn_score[4].get)
total_dist = sum(pfx_mindist[4].values())
total_dist_plot = [total_dist]
print('now \t\t\t\t\t total dist: {}'.format(total_dist))
while (len(sorted_asns) > 0) and (max_monitors > 0):
    best_asn = sorted_asns.pop()
    improvement = asn_score[4][best_asn]
    max_monitors -= 1
    for pfx in candidates.keys():
        if (':' not in pfx) and (best_asn in candidates[pfx].keys()):
            if candidates[pfx][best_asn] < pfx_mindist[4][pfx]:
                for other_asn in candidates[pfx].keys():
                    if other_asn != best_asn:
                        previous_improvement = max(0, pfx_mindist[4][pfx] - candidates[pfx][other_asn])
                        asn_score[4][ other_asn ] -= previous_improvement
                        asn_score[4][ other_asn ] += max(0, candidates[pfx][best_asn] - candidates[pfx][other_asn])
                pfx_mindist[4][pfx] = candidates[pfx][best_asn]
            del candidates[pfx][best_asn]
    del asn_score[4][best_asn]
    sorted_asns = sorted(asn_score[4].keys(), key=asn_score[4].get)
    total_dist -= improvement
    total_dist_plot.append(total_dist)
    print('ASN: {}\t improvement: {}\t total dist: {}'.format(best_asn, improvement, total_dist))

plt.plot(range(MAX_MONITORS+1), total_dist_plot)
plt.show()