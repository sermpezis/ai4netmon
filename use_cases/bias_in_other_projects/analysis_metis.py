import json

INPUT_FILE = './data/{}.v4.2022-10-31.selection.json'
OUTPUT_FILE = './data/metis_lists_{}.json'

SIMILARITY_TYPES = ['as_path_length', 'rtt', 'ip_hops']
NB_PROBES = [100, 200, 500, 1000, 3000]

for st in SIMILARITY_TYPES:
    with open(INPUT_FILE.format(st), 'r') as f:
        data = json.load(f)

    lists_of_networks = dict()
    recommendations = [i['asn'] for i in data['results']]

    for nb in NB_PROBES:
        lists_of_networks[str(nb)] = recommendations[0:nb+1]

    with open(OUTPUT_FILE.format(st), 'w') as f:
        json.dump(lists_of_networks, f)
