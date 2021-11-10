This dataset is created with the use of Javascript. 
In order to run the script we will need:
* Node.js
* npm
Then, we open cmd in the folder Datasets/As-rank, and we run the following command:
* npm install (Only the first time we run the project. Npm install will download the dependencies).
Finally, we run:
* node index.js (In order to run the script).
Dependencies:
* [Library needle](https://github.com/tomas/needle?fbclid=IwAR31rnPqzQJtBp-XWnku2ld5bsaVZiSAurAH1MOEe2XFT632f0bTlND5HKQ).

Specifically, we stored in a global array the below features.
* **AS number**: Each AS has a unique number
* **Rank**:
* **Source**: Company/Organization in which each AS belongs to.
* **Longitude**: A geographic coordinate that specifies the east–west position of a point on the Earth's surface.
* **Latitude**: A geographic coordinate that specifies the north–south position of a point on the Earth's surface.
* **Number of ASns**: Indicates the number of ASns with which an AS exchange information
* **Number of prefixes**: A network prefix is an aggregation of IP addresses.
* **Number of addresses**: Internet addresses represent a network interface. Currently, the Internet runs two protocol versions of IP: version 4 and 6.
* **Iso**: The country where the AS is located.
* **Total**: The sum of customers, peers and providers that an AS communicates with.
* **Customer**: The number of customers.
* **Peer**: The number of peers.
* **Provider**: The number of providers.