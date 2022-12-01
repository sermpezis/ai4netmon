Below we present a bias analysis (radar plot & detailed distributions) for the sets of monitors:

- Recommendations by the Metis system for adding extra probes in RIPE Atlas. There are 3 files for the recommendations of Metis, each corresponding to a different similarity metric. In the files the number of added probes are the keys of the dictionary and the lists of the probes to be added are the values of the dictionary. In total, we will analyze 5 sets of monitors _for each metric_ : the sets will be the sets of **RIPE Atlas probes + list of recommendations**.

- Compare them with (i) the entire set of ASes and (ii) the set of RIPE Atlas probes. 

In total the radar plot will have 5+2 lines, and the distribution plots 5+2 sets of bars.

# RTT metric
Lists of monitors  [./data/metis_lists_rtt.json](./data/metis_lists_rtt.json)

## Radar Plot

## Detailed distributions


# IP HOPS metric
Lists of monitors  [./data/metis_lists_ip_hops.json](./data/metis_lists_ip_hops.json)

## Radar Plot

:-------------------------:
![Radar plot ](./figures/METIS_iphops/fig_radar_all_metis_Atlas_IPhops.png?raw=true) 

## Detailed distributions

**Location related dimensions**

&nbsp;|RIR region|Location (continent)|&nbsp;| &nbsp;
:---:|:---:|:---:|:---:|:---:
&nbsp; |![](./figures/METIS_iphops/Fig_Histogram_AS_rank_source_metis_iphops_lists.png?raw=true)| ![](./figures/METIS_iphops/Fig_Histogram_AS_rank_continent_metis_iphops_lists.png?raw=true)|&nbsp;|&nbsp;


**Network size dimensions**

Customer cone (#ASNs) | Customer cone (#prefixes) | Customer cone (#addresses) | AS hegemony | &nbsp;
:---:|:---:|:---:|:---:|:---:
![](./figures/METIS_iphops/Fig_CDF_AS_rank_numberAsns_metis_iphops_lists.png?raw=true)|![](./figures/METIS_iphops/Fig_CDF_AS_rank_numberPrefixes_metis_iphops_lists.png?raw=true)|![](./figures/METIS_iphops/Fig_CDF_AS_rank_numberAddresses_metis_iphops_lists.png?raw=true)|![](./figures/METIS_iphops/Fig_CDF_AS_hegemony_metis_iphops_lists.png?raw=true)|&nbsp;


**Topology related dimensions**

#neighbors (total)|#neighbors (peers)|#neighbors (customers)|#neighbors (providers)|&nbsp;
:---:|:---:|:---:|:---:|:---:
![](./figures/METIS_iphops/Fig_CDF_AS_rank_total_metis_iphops_lists.png?raw=true)|![](./figures/METIS_iphops/Fig_CDF_AS_rank_peer_metis_iphops_lists.png?raw=true)|![](./figures/METIS_iphops/Fig_CDF_AS_rank_customer_metis_iphops_lists.png?raw=true)|![](./figures/METIS_iphops/Fig_CDF_AS_rank_provider_metis_iphops_lists.png?raw=true)|&nbsp;



**IXP related dimensions**

&nbsp;|#IXPs (PeeringDB)|#facilities (PeeringDB)|Peering policy (PeeringDB)|&nbsp;
:---:|:---:|:---:|:---:|:---:
&nbsp;|![](./figures/METIS_iphops/Fig_CDF_peeringDB_ix_count_metis_iphops_lists.png?raw=true)|![](./figures/METIS_iphops/Fig_CDF_peeringDB_fac_count_metis_iphops_lists.png?raw=true)|![](./figures/METIS_iphops/Fig_Histogram_peeringDB_policy_general_metis_iphops_lists.png?raw=true)|&nbsp;


**Network type dimensions**

Network type (PeeringDB)|Traffic ratio (PeeringDB)|Traffic volume (PeeringDB)|Scope (PeeringDB)|Personal ASN
:---:|:---:|:---:|:---:|:---:
![](./figures/METIS_iphops/Fig_Histogram_peeringDB_info_type_metis_iphops_lists.png?raw=true)|![](./figures/METIS_iphops/Fig_Histogram_peeringDB_info_ratio_metis_iphops_lists.png?raw=true)|![](./figures/METIS_iphops/Fig_Histogram_peeringDB_info_traffic_metis_iphops_lists.png?raw=true)|![](./figures/METIS_iphops/Fig_Histogram_peeringDB_info_scope_metis_iphops_lists.png?raw=true)|![](./figures/METIS_iphops/Fig_Histogram_is_personal_AS_metis_iphops_lists.png?raw=true)



# AS PATH LENGTH metric
Lists of monitors  [./data/metis_lists_as_path_length.json](./data/metis_lists_as_path_length.json)

## Radar Plot
:-------------------------:
![Radar plot](./figures/METIS_as_paths_lengths/fig_radar_all_metis_Atlas_ASpathlen.png?raw=true) 

## Detailed distributions

**Location related dimensions**

&nbsp;|RIR region|Location (continent)|&nbsp;| &nbsp;
:---:|:---:|:---:|:---:|:---:
&nbsp; |![](./figures/METIS_as_paths_lengths/Fig_Histogram_AS_rank_source_metis_aspathlen.png?raw=true)| ![](./figures/METIS_as_paths_lengths/Fig_Histogram_AS_rank_continent_metis_aspathlen.png?raw=true)|&nbsp;|&nbsp;


**Network size dimensions**

Customer cone (#ASNs) | Customer cone (#prefixes) | Customer cone (#addresses) | AS hegemony | &nbsp;
:---:|:---:|:---:|:---:|:---:
![](./figures/METIS_as_paths_lengths/Fig_CDF_AS_rank_numberAsns_metis_aspathlen.png?raw=true)|![](./figures/METIS_as_paths_lengths/Fig_CDF_AS_rank_numberPrefixes_metis_aspathlen.png?raw=true)|![](./figures/METIS_as_paths_lengths/Fig_CDF_AS_rank_numberAddresses_metis_aspathlen.png?raw=true)|![](./figures/METIS_as_paths_lengths/Fig_CDF_AS_hegemony_metis_aspathlen.png?raw=true)|&nbsp;


**Topology related dimensions**

#neighbors (total)|#neighbors (peers)|#neighbors (customers)|#neighbors (providers)|&nbsp;
:---:|:---:|:---:|:---:|:---:
![](./figures/METIS_as_paths_lengths/Fig_CDF_AS_rank_total_metis_aspathlen.png?raw=true)|![](./figures/METIS_as_paths_lengths/Fig_CDF_AS_rank_peer_metis_aspathlen.png?raw=true)|![](./figures/METIS_as_paths_lengths/Fig_CDF_AS_rank_customer_metis_aspathlen.png?raw=true)|![](./figures/METIS_as_paths_lengths/Fig_CDF_AS_rank_provider_metis_aspathlen.png?raw=true)|&nbsp;



**IXP related dimensions**

&nbsp;|#IXPs (PeeringDB)|#facilities (PeeringDB)|Peering policy (PeeringDB)|&nbsp;
:---:|:---:|:---:|:---:|:---:
&nbsp;|![](./figures/METIS_as_paths_lengths/Fig_CDF_peeringDB_ix_count_metis_aspathlen.png?raw=true)|![](./figures/METIS_as_paths_lengths/Fig_CDF_peeringDB_fac_count_metis_aspathlen.png?raw=true)|![](./figures/METIS_as_paths_lengths/Fig_Histogram_peeringDB_policy_general_metis_aspathlen.png?raw=true)|&nbsp;


**Network type dimensions**

Network type (PeeringDB)|Traffic ratio (PeeringDB)|Traffic volume (PeeringDB)|Scope (PeeringDB)|Personal ASN
:---:|:---:|:---:|:---:|:---:
![](./figures/METIS_as_paths_lengths/Fig_Histogram_peeringDB_info_type_metis_aspathlen.png?raw=true)|![](./figures/METIS_as_paths_lengths/Fig_Histogram_peeringDB_info_ratio_metis_aspathlen.png?raw=true)|![](./figures/METIS_as_paths_lengths/Fig_Histogram_peeringDB_info_traffic_metis_aspathlen.png?raw=true)|![](./figures/METIS_as_paths_lengths/Fig_Histogram_peeringDB_info_scope_metis_aspathlen.png?raw=true)|![](./figures/METIS_as_paths_lengths/Fig_Histogram_is_personal_AS_metis_aspathlen.png?raw=true)
