#!/usr/bin/env python3
import sys
from collections import defaultdict
from itertools import groupby
import json
import numpy as np
import pickle


DEFAULT_ROUTES = ['0.0.0.0/0', '::/0']
INPUT_STREAM = sys.stdin
LARGE_PATH_LENGTH = 10000
# AS_to_pfx_dist_FILENAME = 'TEST_AS_to_pfx_dist.json'
AS_to_pfx_dist_FILENAME_PKL = 'TEST_AS_to_pfx_dist.pkl'
AS_to_pfx_dist_mean_FILENAME = 'TEST_AS_to_pfx_dist_mean.json'
LIST_RIPE_MONITORS_FILENAME = 'TEST_LIST_RIPE_MONITORS.json'

def load_data_ripe_bgp_dumps(input_stream):
    bgp_messages = [] 
    for line in input_stream:
        line = line.rstrip('\n')
        fields = line.split('|')

        peer = fields[3]
        peer_asn = fields[4]
        pfx  = fields[5]
        path = fields[6].split(' ')

        yield [peer, peer_asn, pfx, path]

    #     bgp_messages.append([peer, peer_asn, pfx, path])
    # return bgp_messages

def remove_as_sets_from_path(path, how='omit'):
    if how == 'omit':
        return [i for i in path if '{' not in i]
    else:
        raise Exception('This method for handling AS-sets is not defined: '+how)

def remove_duplicates_from_path(path):
    return [i[0] for i in groupby(path)]

def remove_path_poisoning(path):
    start_i = path.index(path[-1])
    return path[0:start_i+1]

def clean_path(path, as_set=True, as_set_type='omit', duplicates=True, poisoning=True):
    if as_set:
        path = remove_as_sets_from_path(path, how=as_set_type)
    if duplicates:
        path = remove_duplicates_from_path(path)
    if poisoning:
        path = remove_path_poisoning(path)
    return path



def main():
    # bgp_messages = load_data_ripe_bgp_dumps(INPUT_STREAM)
    AS_to_pfx_dist = defaultdict(dict)
    set_of_RIPE_monitors = set()
    for msg in load_data_ripe_bgp_dumps(INPUT_STREAM):
        peer_asn = msg[1]
        pfx = msg[2]
        path = clean_path(msg[3])
        
        if pfx in DEFAULT_ROUTES:
            continue
        set_of_RIPE_monitors.add(peer_asn)
        
        for i, AS in enumerate(path):
            current_dist = len(path) - i -1
            AS_to_pfx_dist[AS][pfx] = min(current_dist, AS_to_pfx_dist[AS].get(pfx, LARGE_PATH_LENGTH))
    
    with open(LIST_RIPE_MONITORS_FILENAME, 'w') as f:
        json.dump(list(set_of_RIPE_monitors), f)

    # with open(AS_to_pfx_dist_FILENAME, 'w') as f:
    #     json.dump(AS_to_pfx_dist, f)

    AS_to_pfx_dist_mean = dict()
    for AS in AS_to_pfx_dist.keys():
        AS_to_pfx_dist_mean[AS] = np.mean(list(AS_to_pfx_dist[AS].values()))
    with open(AS_to_pfx_dist_mean_FILENAME, 'w') as f:
        json.dump(AS_to_pfx_dist_mean, f)

    with open(AS_to_pfx_dist_FILENAME_PKL, 'wb') as f:
        pickle.dump(AS_to_pfx_dist, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()