We present an analysis of the characteristics of the networks that are "responsible", "true", "wrong", and "origin" in the list of broken links dataset.

- Source data: In the file [./data/broken_links_lists.json](./data/broken_links_lists.json) there are 4 lists of monitors ('responsible', 'true', 'wrong', 'origin'), extracted from the initial file of broken links.


In particular, we compare each of the above sets with the entire set of ASes, and calculate their (i) bias (see radar plot below) and (ii) detailed distibutions (see CDFs and Histograms below).

**Bias (radar plot)**: Easily depicts which set is more different than the entire set of ASes, and at which dimension. A set that is very different that the entire set of ASes at a dimension X (e.g., Network type) has a value that is far from the center in the radar plot. The farther the more different. The plot can be used for getting quick insights before delving into details in the distribution plots.

**Detailed distribution plots**: Depict the distribution of the characteristics of the different sets, and the distribution of the entire population of ASes. For variables that take continuous values (e.g., connected to number of IXPs, or number of ASes in the customer cone) we use CDF plots, and for categorical variables (e.g., network type, or continent) we use histograms. These plots can be used to get insights for what types of networks are typically in a set (e.g., there are many NSPs in the "true" networks).
- In the CDFs: (i) The farther a curve is from the "All ASes" curve, the more the set differs from the entire population. (ii) Curves that are more at the right, denote a distribution of higher values (e.g., in the #IXPs, a curve A on the right of a curve B, denotes that the networks in the set A tend to be connected to more IXPs than the networks in the set B)
- In the Histogram plots, the more different the height of a bar is from the bar of the "All ASes", the more different the set is from the entire population.
- Clicking at a plot, opens the plot in a different page (larger size)

## Radar Plot
bias - radar plot
:-------------------------:
![Radar plot - ris_rv_ris+rv_bgp_bias](./figures/broken_lists/fig_radar_all_broken.png?raw=true) 

## Detailed distributions

**Location related dimensions**

&nbsp;|RIR region|Location (continent)|&nbsp;| &nbsp;
:---:|:---:|:---:|:---:|:---:
&nbsp; |![](./figures/broken_lists/Fig_Histogram_AS_rank_source_broken_lists.png?raw=true)| ![](./figures/broken_lists/Fig_Histogram_AS_rank_continent_broken_lists.png?raw=true)|&nbsp;|&nbsp;


**Network size dimensions**

Customer cone (#ASNs) | Customer cone (#prefixes) | Customer cone (#addresses) | AS hegemony | &nbsp;
:---:|:---:|:---:|:---:|:---:
![](./figures/broken_lists/Fig_CDF_AS_rank_numberAsns_broken_lists.png?raw=true)|![](./figures/broken_lists/Fig_CDF_AS_rank_numberPrefixes_broken_lists.png?raw=true)|![](./figures/broken_lists/Fig_CDF_AS_rank_numberAddresses_broken_lists.png?raw=true)|![](./figures/broken_lists/Fig_CDF_AS_hegemony_broken_lists.png?raw=true)|&nbsp;


**Topology related dimensions**

#neighbors (total)|#neighbors (peers)|#neighbors (customers)|#neighbors (providers)|&nbsp;
:---:|:---:|:---:|:---:|:---:
![](./figures/broken_lists/Fig_CDF_AS_rank_total_broken_lists.png?raw=true)|![](./figures/broken_lists/Fig_CDF_AS_rank_peer_broken_lists.png?raw=true)|![](./figures/broken_lists/Fig_CDF_AS_rank_customer_broken_lists.png?raw=true)|![](./figures/broken_lists/Fig_CDF_AS_rank_provider_broken_lists.png?raw=true)|&nbsp;



**IXP related dimensions**

&nbsp;|#IXPs (PeeringDB)|#facilities (PeeringDB)|Peering policy (PeeringDB)|&nbsp;
:---:|:---:|:---:|:---:|:---:
&nbsp;|![](./figures/broken_lists/Fig_CDF_peeringDB_ix_count_broken_lists.png?raw=true)|![](./figures/broken_lists/Fig_CDF_peeringDB_fac_count_broken_lists.png?raw=true)|![](./figures/broken_lists/Fig_Histogram_peeringDB_policy_general_broken_lists.png?raw=true)|&nbsp;


**Network type dimensions**

Network type (PeeringDB)|Traffic ratio (PeeringDB)|Traffic volume (PeeringDB)|Scope (PeeringDB)|Personal ASN
:---:|:---:|:---:|:---:|:---:
![](./figures/broken_lists/Fig_Histogram_peeringDB_info_type_broken_lists.png?raw=true)|![](./figures/broken_lists/Fig_Histogram_peeringDB_info_ratio_broken_lists.png?raw=true)|![](./figures/broken_lists/Fig_Histogram_peeringDB_info_traffic_broken_lists.png?raw=true)|![](./figures/broken_lists/Fig_Histogram_peeringDB_info_scope_broken_lists.png?raw=true)|![](./figures/broken_lists/Fig_Histogram_is_personal_AS_broken_lists.png?raw=true)

