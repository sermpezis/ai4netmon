import json
import requests

OUTPUT_FILE = './data/mvp_lists.json'


API_URL = "http://5.161.124.63/mvp?volume={}GB"

VOLUMES = [1,5,10,20,100]


lists_of_networks = dict()

for v in VOLUMES:
    r = requests.get(API_URL.format(v))
    data = r.json()
    lists_of_networks[str(v)+'GB'] = list(set([int(i.split('_')[-1]) for i in data['VP_set']]))

with open(OUTPUT_FILE, 'w') as f:
    json.dump(lists_of_networks, f)
