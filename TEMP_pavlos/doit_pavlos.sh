YYYY=2021
MM=07
DD=01

THRES=28 # max number of peers per ASN

echo "creating peer.pfx.pathlen_full.$YYYY.$MM.$DD.txt.gz"
if [ ! -f peer.pfx.pathlen_full.$YYYY.$MM.$DD.txt.gz ]; then
   for RRC in RRC in {00..26}
   do 
      bgpdump -m -t change ./BGPdumps/rrc$RRC.bview.$YYYY$MM$DD.0000.gz
   done | ./pathlen_pavlos.py | gzip -9 > peer.pfx.pathlen_full.$YYYY.$MM.$DD.txt.gz
fi


### creates a file with
# peers/prefixes (count) (min dist) (max dist)
echo "creating stats_no_country.$YYYY.$MM.$DD.txt.gz"
if [ ! -f stats_no_country.$YYYY.$MM.$DD.txt.gz ]; then
   ./dist-stats_pavlos.py peer.pfx.pathlen_full.$YYYY.$MM.$DD.txt.gz $YYYY-$MM-$DD | gzip -9 > stats_no_country.$YYYY.$MM.$DD.txt.gz
fi


### improvements files has
## address family ASN improvement factor (sum AS-hops per peer)
echo "creating improvements_no_country.$YYYY.$MM.$DD.txt"
if [ ! -f improvements_no_country.$YYYY.$MM.$DD.txt.gz ]; then
   for RRC in RRC in {00..26}
   do 
      bgpdump -m -t change ./BGPdumps/rrc$RRC.bview.$YYYY$MM$DD.0000.gz
   done | ./calc-dist-improvements_pavlos.py peer.pfx.pathlen_full.$YYYY.$MM.$DD.txt.gz stats_no_country.$YYYY.$MM.$DD.txt.gz $THRES $YYYY-$MM-$DD | tee improvements_no_country.$YYYY.$MM.$DD.txt
fi
