import json
import requests

OUTPUT_FILE = './data/periscope_lists.json'


API_URL = "https://api.periscope.caida.org/v2/host/list?command={}"

COMMAND = ["traceroute", "ping", "bgp"]


lists_of_networks = dict()

for cm in COMMAND:
    r = requests.get(API_URL.format(cm))
    data = r.json()
    lists_of_networks[cm] = list(set([d["asn"] for d in data]))

with open(OUTPUT_FILE, 'w') as f:
    json.dump(lists_of_networks, f)
