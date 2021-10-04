import pybgpstream
import networkx as nx
from collections import defaultdict
from itertools import groupby

# Create an instance of a simple undirected graph
as_graph = nx.Graph()

bgp_lens = defaultdict(lambda: defaultdict(lambda: None))

stream = pybgpstream.BGPStream(
    # Consider this time interval:
    # Sat, 01 Aug 2015 7:50:00 GMT -  08:10:00 GMT
    from_time="2015-08-01 07:50:00", until_time="2015-08-01 08:10:00",
    collectors=["rrc00"],
    record_type="ribs",
)

for rec in stream.records():
    for elem in rec:
        # Get the peer ASn
        peer = str(elem.peer_asn)
        # Get the array of ASns in the AS path and remove repeatedly prepended ASns
        hops = [k for k, g in groupby(elem.fields['as-path'].split(" "))]
        if len(hops) > 1 and hops[0] == peer:
            # Get the origin ASn
            origin = hops[-1]
            # Add new edges to the NetworkX graph
            for i in range(0,len(hops)-1):
                as_graph.add_edge(hops[i],hops[i+1])
            # Update the AS path length between 'peer' and 'origin'
            bgp_lens[peer][origin] = \
                min(list(filter(bool,[bgp_lens[peer][origin],len(hops)])))

# For each 'peer' and 'origin' pair
for peer in bgp_lens:
    for origin in bgp_lens[peer]:
        # compute the shortest path in the NetworkX graph
        nxlen = len(nx.shortest_path(as_graph, peer, origin))
        # and compare it to the BGP hop length
        print((peer, origin, bgp_lens[peer][origin], nxlen))
