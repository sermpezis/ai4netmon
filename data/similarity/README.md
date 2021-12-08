# Data related to network/monitor similarity. 

The naming of the datasets are as follows: `<set>_<distance or similarity>_<method>_<date[YYYYMMDD]>.csv`. 
The _first row_ and _first column_ of the csv files should correspond to the network/monitor IDs (and the first field of the first row to be empty) so that they can be loaded with `pandas.read_csv(filename, header=0, index_col=0)`.
More details for each dataset below.

* **`ripe_ris_distance_pathlens_100k_20210701.csv`**: Distances between RIPE RIS peers (IDs used: Peer IP). Distance between two peers is calculated based on the number of AS-hops in paths from RIPE RIS RRCs for a random sample of 100k prefixes. Specifically, if a peer X has paths of length `[p1, p2, ...]` for a set of prefixes and another peer Y has paths of length `[q1, q2, ...]` for the same prefixes, then the distance of X,Y is calculated as `dist(X,Y) = euclidean_distance([p1,p2,...], [q1, q2, ...]) / #prefixes`


* **`ripe_atlas_probe_asn_similarity_jaccard_paths_v4_median75_asn_max_20211124.csv`**: [link to download - large file]( https://drive.google.com/file/d/1bLUBHYe0I66w0n3RFA9F5bfFQI9AIg9A/view?usp=sharing ) Similarities between RIPE Atlas probes (IDs used: ASNs hosting the probes). Similarities calculated based on the jaccard index of the common ASes in traceroute paths to the same destinations (see [paper]( https://clarinet.u-strasbg.fr/~pelsser/publications/Holterbach-similarity-anrw17.pdf ); used the 'median75' field (i.e., the 75th percentile of jaccard), and for the granularity at the ASN level took the max value among all correspondins probe pairs (e.g., if `p1` in `ASNx` and `p2` and `p3` in `ASNy`, then the `similarity(ASN1,ASN2)` is the maximum of `similarity(p1,p2)` and `similarity(p1,p3)`).