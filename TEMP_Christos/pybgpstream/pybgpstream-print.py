#!/usr/bin/env python

import pybgpstream
stream = pybgpstream.BGPStream(
    from_time="2017-07-07 00:00:00", until_time="2017-07-07 00:10:00 UTC",
    collectors=["route-views.sg", "route-views.eqix"],
    record_type="updates",
    filter="peer 11666 and prefix more 210.180.0.0/16"
)

for elem in stream:
    # record fields can be accessed directly from elem
    # e.g. elem.time
    # or via elem.record
    # e.g. elem.record.time
    print(elem)