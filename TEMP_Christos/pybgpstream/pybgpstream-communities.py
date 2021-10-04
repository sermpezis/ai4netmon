#!/usr/bin/env python

import pybgpstream
from collections import defaultdict

stream = pybgpstream.BGPStream(
    # Consider this time interval:
    # Sat, 01 Aug 2015 7:50:00 GMT -  08:10:00 GMT
    from_time="2015-08-01 07:50:00", until_time="2015-08-01 08:10:00",
    collectors=["rrc06"],
    record_type="ribs",
    filter="peer 25152 and prefix more 185.84.166.0/23 and community *:3400"
)

# <community, prefix > dictionary
community_prefix = defaultdict(set)

# Get next record
for rec in stream.records():
    for elem in rec:
        # Get the prefix
        pfx = elem.fields['prefix']
        # Get the associated communities
        communities = elem.fields['communities']
        # for each community save the set of prefixes
        # that are affected
        for c in communities:
            community_prefix[c].add(pfx)

# Print the list of MOAS prefix and their origin ASns
for ct in community_prefix:
    print("Community:", ct, "==>", ",".join(community_prefix[ct]))
1