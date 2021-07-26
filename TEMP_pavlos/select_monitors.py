#!/usr/bin/env python3
import json
import gzip
from matplotlib import pyplot as plt
from collections import defaultdict

INF = 1000000

YYYY='2021'
MM='07'
DD='01'

DIST_FILENAME = 'pfx.asn.min_distance{}.{}.{}.json'.format(YYYY,MM,DD)
STATS_FILENAME = 'stats_no_country.{}.{}.{}.txt.gz'.format(YYYY,MM,DD)

THRES = 1
MAX_MONITORS = 10000



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
sorted_asns = sorted(asn_score[4].keys(), key=asn_score[4].get)


## naive
total_dist = sum(pfx_mindist[4].values())
total_dist_plot_naive = [total_dist]

selected_asns_naive = sorted_asns[-MAX_MONITORS:]
selected_asns_naive = selected_asns_naive[::-1]
min_dist_naive = defaultdict()
for pfx in candidates.keys():    
        if (':' not in pfx):
            min_dist_naive[pfx] = pfx_mindist[4][pfx]
for i,next_asn in enumerate(selected_asns_naive):
    print(i)
    for pfx in candidates.keys():    
        if (':' not in pfx):
            # min_dist_naive[pfx]  = min([min_dist_naive[pfx], candidates[pfx].get(next_asn,INF)])
            # improvement = max(0, pfx_mindist[4][pfx]- min_dist_naive[pfx])

            improvement = max(0, min_dist_naive[pfx]-candidates[pfx].get(next_asn,INF))
            min_dist_naive[pfx] -= improvement

            # min_dist_naive = min([candidates[pfx][asn] for asn in selected_asns_naive[0:i+1] if asn in candidates[pfx].keys()]+[INF])
            # improvement = max(0, pfx_mindist[4][pfx]- min_dist_naive)
            total_dist -= improvement
    total_dist_plot_naive.append(total_dist)

print(selected_asns_naive)
print(total_dist_plot_naive)

## greedy
# total_dist = sum(pfx_mindist[4].values())
# total_dist_plot = [total_dist]
# selected_asns = []
# print('now \t\t\t\t\t total dist: {}'.format(total_dist))
# while (len(sorted_asns) > 0) and (len(selected_asns) <MAX_MONITORS):
#     best_asn = sorted_asns.pop()
#     selected_asns.append(selected_asns)
#     improvement = asn_score[4][best_asn]
#     for pfx in candidates.keys():
#         if (':' not in pfx) and (best_asn in candidates[pfx].keys()):
#             if candidates[pfx][best_asn] < pfx_mindist[4][pfx]:
#                 for other_asn in candidates[pfx].keys():
#                     if other_asn != best_asn:
#                         previous_improvement = max(0, pfx_mindist[4][pfx] - candidates[pfx][other_asn])
#                         asn_score[4][ other_asn ] -= previous_improvement
#                         asn_score[4][ other_asn ] += max(0, candidates[pfx][best_asn] - candidates[pfx][other_asn])
#                 pfx_mindist[4][pfx] = candidates[pfx][best_asn]
#             del candidates[pfx][best_asn]
#     del asn_score[4][best_asn]
#     sorted_asns = sorted(asn_score[4].keys(), key=asn_score[4].get)
#     total_dist -= improvement
#     total_dist_plot.append(total_dist)
#     print('ASN: {}\t improvement: {}\t total dist: {}'.format(best_asn, improvement, total_dist))
# print(total_dist_plot)



## greedy v2
total_dist = sum(pfx_mindist[4].values())
total_dist_plot = [total_dist]
selected_asns = []
min_dist_greedy = defaultdict(lambda : INF)
print('now \t\t\t\t\t total dist: {}'.format(total_dist))
while (len(sorted_asns) > 0) and (len(selected_asns) <MAX_MONITORS):
    best_asn = sorted_asns.pop()
    selected_asns.append(best_asn)
    # improvement = asn_score[4][best_asn]
    for pfx in candidates.keys():
        if (':' not in pfx) and (best_asn in candidates[pfx].keys()):
            if candidates[pfx][best_asn] < pfx_mindist[4][pfx]:
                improvement = pfx_mindist[4][pfx] - candidates[pfx][best_asn]
                total_dist -= improvement
                for other_asn in candidates[pfx].keys():
                    if other_asn != best_asn:
                        previous_improvement = max(0, pfx_mindist[4][pfx] - candidates[pfx][other_asn])
                        asn_score[4][ other_asn ] -= previous_improvement
                        asn_score[4][ other_asn ] += max(0, candidates[pfx][best_asn] - candidates[pfx][other_asn])
                pfx_mindist[4][pfx] = candidates[pfx][best_asn]
            del candidates[pfx][best_asn]
    del asn_score[4][best_asn]
    sorted_asns = sorted(asn_score[4].keys(), key=asn_score[4].get)
    # total_dist -= improvement
    total_dist_plot.append(total_dist)
    print('ASN: {}\t improvement: {}\t total dist: {}'.format(best_asn, improvement, total_dist))
print(total_dist_plot)



# plot results
plt.plot(range(MAX_MONITORS+1), total_dist_plot, range(MAX_MONITORS+1),total_dist_plot_naive)
plt.legend(['Greedy', 'Sorted'])
plt.grid('on')
plt.xlabel('# new monitors')
plt.ylabel('total distance')
plt.savefig('set_monitors_improvement_{}.png'.format(MAX_MONITORS))
# plt.show()



# D = defaultdict(list)
# D['asns naive'] = selected_asns_naive
# D['dist naive'] = total_dist_plot_naive
# D['asns greedy'] = selected_asns
# D['dist greedy'] = total_dist_plot
D = {'asns naive': selected_asns_naive, 'dist naive': total_dist_plot_naive, 'asns greedy': selected_asns, 'dist greedy': total_dist_plot}
print(selected_asns)
print(D)
with open('select_monitors_{}.json'.format(MAX_MONITORS), 'w') as f:
    json.dump(D,f)