The file contains the AS-graph in the Internet in the form of an edgelist.

The lines starting with '#' are comments; can be omitted.

Each line corresponds to an AS-link, including the two ASes between the link exists and the inferred AS-relationship between these two ASes.
The as-rel files contain p2p (peer-to-peer; denoted with "0") and p2c (provider-to-customer; denoted with "-1") relationships.  The format is:
<provider-as>|<customer-as>|-1
<peer-as>|<peer-as>|0|<source>
