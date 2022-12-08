Below we present a bias analysis (radar plot & detailed distributions) for the sets of monitors:

- Looking glasses in the [CAIDA Periscope platform](https://www.caida.org/catalog/software/looking-glass-api). In the file [./data/periscope_lists.json](./data/periscope_lists.json) the types of looking glasses are the keys of the dictionary and the lists of the corresponding looking glasses the values of the dictionary.

- Compare them with (i) RIPE RIS (ii) RouteViews (iii) RIPE Atlas


## Radar Plot

![Radar plot - ris_rv_atlas_periscope_bias](./figures/periscope/fig_radar_all_ris_rv_atlas_periscope.png?raw=true) 


## Detailed distributions

&nbsp;|RIR region|Location (continent)|&nbsp;| &nbsp;
:---:|:---:|:---:|:---:|:---:
&nbsp; |![](./figures/periscope/Fig_Histogram_AS_rank_source_periscope_lists.png?raw=true)| ![](./figures/periscope/Fig_Histogram_AS_rank_continent_periscope_lists.png?raw=true)|&nbsp;|&nbsp;


**Network size dimensions**

Customer cone (#ASNs) | Customer cone (#prefixes) | Customer cone (#addresses) | AS hegemony | &nbsp;
:---:|:---:|:---:|:---:|:---:
![](./figures/periscope/Fig_CDF_AS_rank_numberAsns_periscope_lists.png?raw=true)|![](./figures/periscope/Fig_CDF_AS_rank_numberPrefixes_periscope_lists.png?raw=true)|![](./figures/periscope/Fig_CDF_AS_rank_numberAddresses_periscope_lists.png?raw=true)|![](./figures/periscope/Fig_CDF_AS_hegemony_periscope_lists.png?raw=true)|&nbsp;


**Topology related dimensions**

#neighbors (total)|#neighbors (peers)|#neighbors (customers)|#neighbors (providers)|&nbsp;
:---:|:---:|:---:|:---:|:---:
![](./figures/periscope/Fig_CDF_AS_rank_total_periscope_lists.png?raw=true)|![](./figures/periscope/Fig_CDF_AS_rank_peer_periscope_lists.png?raw=true)|![](./figures/periscope/Fig_CDF_AS_rank_customer_periscope_lists.png?raw=true)|![](./figures/periscope/Fig_CDF_AS_rank_provider_periscope_lists.png?raw=true)|&nbsp;



**IXP related dimensions**

&nbsp;|#IXPs (PeeringDB)|#facilities (PeeringDB)|Peering policy (PeeringDB)|&nbsp;
:---:|:---:|:---:|:---:|:---:
&nbsp;|![](./figures/periscope/Fig_CDF_peeringDB_ix_count_periscope_lists.png?raw=true)|![](./figures/periscope/Fig_CDF_peeringDB_fac_count_periscope_lists.png?raw=true)|![](./figures/periscope/Fig_Histogram_peeringDB_policy_general_periscope_lists.png?raw=true)|&nbsp;


**Network type dimensions**

Network type (PeeringDB)|Traffic ratio (PeeringDB)|Traffic volume (PeeringDB)|Scope (PeeringDB)|Personal ASN
:---:|:---:|:---:|:---:|:---:
![](./figures/periscope/Fig_Histogram_peeringDB_info_type_periscope_lists.png?raw=true)|![](./figures/periscope/Fig_Histogram_peeringDB_info_ratio_periscope_lists.png?raw=true)|![](./figures/periscope/Fig_Histogram_peeringDB_info_traffic_periscope_lists.png?raw=true)|![](./figures/periscope/Fig_Histogram_peeringDB_info_scope_periscope_lists.png?raw=true)|![](./figures/periscope/Fig_Histogram_is_personal_AS_periscope_lists.png?raw=true)
