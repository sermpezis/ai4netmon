#!/usr/bin/env python3
import json 
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import pickle

AS_to_pfx_dist_FILENAME = 'TEST_AS_to_pfx_dist.json'
AS_to_pfx_dist_FILENAME_PKL = 'TEST_AS_to_pfx_dist.pkl'
# AS_to_pfx_dist_mean_FILENAME = 'TEST_AS_to_pfx_dist_mean.json'
LIST_RIPE_MONITORS_FILENAME = 'TEST_LIST_RIPE_MONITORS.json'

with open(LIST_RIPE_MONITORS_FILENAME, 'r') as f:
    list_of_RIPE_monitors = json.load(f)
# with open(AS_to_pfx_dist_FILENAME, 'r') as f:
#     AS_to_pfx_dist = json.load(f)
with open(AS_to_pfx_dist_FILENAME_PKL, 'rb') as f:
    AS_to_pfx_dist = pickle.load(f)

all_ASNs = set(AS_to_pfx_dist.keys())
non_monitor_ASNs = list(all_ASNs - set(list_of_RIPE_monitors))
monitor_ASNs = list(all_ASNs.intersection(set(list_of_RIPE_monitors)))

AS_mean_dist = dict()
for AS in AS_to_pfx_dist.keys():
    AS_mean_dist[AS] = np.mean(list(AS_to_pfx_dist[AS].values()))


AS_mean_dist__RIPE = [AS_mean_dist[AS] for AS in monitor_ASNs]
AS_mean_dist__nonRIPE = [AS_mean_dist[AS] for AS in non_monitor_ASNs]
print(len(monitor_ASNs))
print(len(non_monitor_ASNs))
print()
ecdf_ripe = ECDF(AS_mean_dist__RIPE)
ecdf_non_ripe = ECDF(AS_mean_dist__nonRIPE)
plt.plot(ecdf_ripe.x, ecdf_ripe.y, '-k', ecdf_non_ripe.x, ecdf_non_ripe.y, '--k')
plt.show()