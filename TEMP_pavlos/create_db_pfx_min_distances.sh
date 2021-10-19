YYYY=2021
MM=07
DD=01

for RRC in RRC in {00..26}
do 
   bgpdump -m -t change ./BGPdumps/rrc$RRC.bview.$YYYY$MM$DD.0000.gz
done | python3 create_db_pfx_min_distances.py