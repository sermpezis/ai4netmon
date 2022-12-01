Below we present a bias analysis (radar plot & detailed distributions) for the sets of monitors:

- Selected monitors by MVP for different volumes V. In the file [./data/mvp_lists.json](./data/mvp_lists.json) the volumes are the keys of the dictionary and the lists of the monitors the values of the dictionary. In total, we will analyze 5 sets of monitors.

- Compare them with (i) the entire set of ASes and (ii) the set of RIPE RIS + RouteViews monitors. 

In total the radar plot will have 5+2 lines, and the distribution plots 5+2 sets of bars.

## Radar Plot

![Radar plot - ris_rv_ris+rv_bgp_bias](./figures/MVP_lists/fig_radar_all_mvp_ris+rv.png?raw=true) 

## Detailed distributions

**Location related dimensions**

&nbsp;|RIR region|Location (continent)|&nbsp;| &nbsp;
:---:|:---:|:---:|:---:|:---:
&nbsp; |![](./figures/MVP_lists/Fig_Histogram_AS_rank_source_mvp_lists.png?raw=true)| ![](./figures/MVP_lists/Fig_Histogram_AS_rank_continent_mvp_lists.png?raw=true)|&nbsp;|&nbsp;


**Network size dimensions**

Customer cone (#ASNs) | Customer cone (#prefixes) | Customer cone (#addresses) | AS hegemony | &nbsp;
:---:|:---:|:---:|:---:|:---:
![](./figures/MVP_lists/Fig_CDF_AS_rank_numberAsns_mvp_lists.png?raw=true)|![](./figures/MVP_lists/Fig_CDF_AS_rank_numberPrefixes_mvp_lists.png?raw=true)|![](./figures/MVP_lists/Fig_CDF_AS_rank_numberAddresses_mvp_lists.png?raw=true)|![](./figures/MVP_lists/Fig_CDF_AS_hegemony_mvp_lists.png?raw=true)|&nbsp;


**Topology related dimensions**

#neighbors (total)|#neighbors (peers)|#neighbors (customers)|#neighbors (providers)|&nbsp;
:---:|:---:|:---:|:---:|:---:
![](./figures/MVP_lists/Fig_CDF_AS_rank_total_mvp_lists.png?raw=true)|![](./figures/MVP_lists/Fig_CDF_AS_rank_peer_mvp_lists.png?raw=true)|![](./figures/MVP_lists/Fig_CDF_AS_rank_customer_mvp_lists.png?raw=true)|![](./figures/MVP_lists/Fig_CDF_AS_rank_provider_mvp_lists.png?raw=true)|&nbsp;



**IXP related dimensions**

&nbsp;|#IXPs (PeeringDB)|#facilities (PeeringDB)|Peering policy (PeeringDB)|&nbsp;
:---:|:---:|:---:|:---:|:---:
&nbsp;|![](./figures/MVP_lists/Fig_CDF_peeringDB_ix_count_mvp_lists.png?raw=true)|![](./figures/MVP_lists/Fig_CDF_peeringDB_fac_count_mvp_lists.png?raw=true)|![](./figures/MVP_lists/Fig_Histogram_peeringDB_policy_general_mvp_lists.png?raw=true)|&nbsp;


**Network type dimensions**

Network type (PeeringDB)|Traffic ratio (PeeringDB)|Traffic volume (PeeringDB)|Scope (PeeringDB)|Personal ASN
:---:|:---:|:---:|:---:|:---:
![](./figures/MVP_lists/Fig_Histogram_peeringDB_info_type_mvp_lists.png?raw=true)|![](./figures/MVP_lists/Fig_Histogram_peeringDB_info_ratio_mvp_lists.png?raw=true)|![](./figures/MVP_lists/Fig_Histogram_peeringDB_info_traffic_mvp_lists.png?raw=true)|![](./figures/MVP_lists/Fig_Histogram_peeringDB_info_scope_mvp_lists.png?raw=true)|![](./figures/MVP_lists/Fig_Histogram_is_personal_AS_mvp_lists.png?raw=true)

