import pybgpstream
import networkx as nx
from collections import defaultdict
from itertools import groupby


stream = pybgpstream.BGPStream(
    # Consider this time interval:
    # Sat, 01 Aug 2015 7:50:00 GMT -  08:10:00 GMT
    # Take ribs(not the updates) for all prefixes for all collectors (RIPE and RouteViews)
    from_time="2021-07-01 07:50:00", until_time="2021-07-01 08:10:00",
    # collectors=["rrc00", "rrc02"],
    record_type="ribs",
)

# Print in a csv file: Collector, pfx, peer_asn, as-path, length of as-path
for elem in stream:
    as_path = elem.fields["as-path"]
    ases = elem.fields["as-path"].split(" ")
    print("Collector: " + elem.collector, "Prefix: " + elem.fields["prefix"], "Peer_asn: " + str(elem.peer_asn), "AS-path: " + as_path, "Length: " + str(len(ases)))
