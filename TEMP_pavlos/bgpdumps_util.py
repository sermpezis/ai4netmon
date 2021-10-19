#!/usr/bin/env python3
from itertools import groupby


def load_data_ripe_bgp_dumps(input_stream):
    bgp_messages = [] 
    for line in input_stream:
        line = line.rstrip('\n')
        fields = line.split('|')

        peer = fields[3]
        peer_asn = fields[4]
        pfx  = fields[5]
        path = fields[6].split(' ')

        yield (peer, peer_asn, pfx, path)


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