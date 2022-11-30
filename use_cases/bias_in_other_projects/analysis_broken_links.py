import json
from collections import defaultdict

INPUT_FILE = './data/broken_links_information.json'
OUTPUT_FILE = './data/broken_links_lists.json'


with open(INPUT_FILE, 'r') as f:
    data = json.load(f)



lists_of_networks = defaultdict(set)
for i in data:
    for d in i:
        for k in d.keys():
            if '_' not in k:
                lists_of_networks[k].add(d[k])

for k,v in lists_of_networks.items():
    lists_of_networks[k] = list(v)
    print(k)
    print(list(v))
    print()




with open(OUTPUT_FILE, 'w') as f:
    json.dump(lists_of_networks, f)