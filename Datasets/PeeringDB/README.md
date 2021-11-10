The dataset (**peeringdb_2_dump_2021_07_01.json**) contains the PeeringDB's snapshots provided by CAIDA.

* **Info ratio**: Accepts 6 values: Balanced, Not Disclosed, Mostly Inbound, Mostly Outbound, Heavy Inbound, Heavy Outbound.
* **Info traffic**: Contains the speed of each Network.
* **Info scope**: Contains info about the continent that the AS is located.
* **Info type**: Gives info about the type of the AS (example: The AS belongs to Government).
* **Info prefixes4**: The number of IPv4 prefixes that are owned by the AS.
* **Info prefixes6**: The number of IPv6 prefixes that are owned by the AS.
* **Policy general**: Contains 4 possible values: Open, Selective, Restrictive and No.
* **Internet exchange count**: Count of internet exchange points. IXPs are generally located at places with pre-existing connections to multiple distinct networks, i.e., Datacenters, and operate physical infrastructure to connect their participants 
* **Facility count**: Counts the facilities/Datacenters.
* **Created**: Timestamp containing the date that the AS number created.

The dataset (**netixlan.json**) contains information about the way that the Internet exchanges network information.
We use this dataset in order to create a bi-graph. For the construction of the graph we use 2 features (ixlan_id, asn).
* **ixlan_id**: The Internet exchange network information (Integer).
* **asn**: The Autonomous system number (Integer).