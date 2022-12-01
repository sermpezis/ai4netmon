Below we present a bias analysis (radar plot & detailed distributions) for the sets of monitors:

- Networks where differences appear. In the file [./data/broken_links_lists.json](./data/broken_links_lists.json) there are 4 lists of monitors ('responsible', 'true', 'wrong', 'origin').

- Compare them with the entire set of ASes. 

In total the radar plot will have 4+1 lines, and the distribution plots 4+1 sets of bars.

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
![](./figures/broken_lists/Fig_CDF_AS_rank_numberAsns_broken_lists.png?raw=true)|![](./figures/broken_lists/Fig_CDF_AS_rank_numberPrefixes_broken_lists.png?raw=true)|![](./figures/broken_lists_Fig_CDF_AS_rank_numberAddresses_broken_lists.png?raw=true)|![](./figures/broken_lists/Fig_CDF_AS_hegemony_broken_lists.png?raw=true)|&nbsp;


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

