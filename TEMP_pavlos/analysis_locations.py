#!/usr/bin/env python3
#
# Author: Pavlos Sermpezis (https://sites.google.com/site/pavlossermpezis/)
#

import pandas as pd
import json
import numpy as np

# df = pd.read_csv('../Datasets/As-rank/asns.csv', sep=",")
# df = df.set_index('asn')

with open('../TEMP - do not push/ris-peers.guessed-and-geocoded.json', 'r') as f:
    ris_data = json.load(f)

rrcs = set([d['rrc'] for d in ris_data])
print(rrcs)
