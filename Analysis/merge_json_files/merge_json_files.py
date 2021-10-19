import pandas as pd
import json
import glob

result = []
for f in glob.glob("merged_file*_*.json"):
    with open(f, "r") as infile:
        result.extend(json.load(infile))

with open("merged_file.json", "w") as outfile:
     json.dump(result, outfile)
