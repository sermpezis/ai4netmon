YYYY=2021
MM=07
DD=01

THRES=28 # max number of peers per ASN

for RRC in RRC in {00..26}
do 
   bgpdump -m -t change ./BGPdumps/rrc$RRC.bview.$YYYY$MM$DD.0000.gz
done | python3 create_list_RIPE_RIS_peers.py