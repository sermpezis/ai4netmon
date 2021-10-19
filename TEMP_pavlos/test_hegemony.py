#!/usr/bin/env python3
#
# Author: Pavlos Sermpezis (https://sites.google.com/site/pavlossermpezis/)
#
from ihr.hegemony import Hegemony
from time import time

oasn = 13335#13335#15169#2914#2501
hege1 = Hegemony(originasns=[oasn], start="2021-10-10 00:00", end="2021-10-10 00:00")
hege2 = Hegemony(originasns=[oasn], start="2021-10-10 10:00", end="2021-10-10 10:00")


t0 = time()
r1 = [r for r in hege1.get_results()]
t1 = time()
r2 = [r for r in hege2.get_results()]
t2 = time()
print(t1-t0)
print(t2-t1)

u1 = set([r['asn'] for r in r1[0]])
u2 = set([r['asn'] for r in r2[0]])
print(len(r1[0]),len(r2[0]))
print(len(u1),len(u2))
# print(r1[0][0])
# print(r2[0][0])
# print(r2[0])