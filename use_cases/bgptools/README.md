#### Table of Contents  
- [Results: Overview - Table with bias values](#Overview-Table-with-bias-values)  
- [Results: Bias between RIPE RIS & Routeviews & RIPE RIS + Routeviews & BGPtools](#ris_rv_ris+rv_bgp-bias)
- [Results: Detailed distributions](#results-detailed-distributions)
- [Results: Bias in BGPtools detailed](#bgp-detailed-bias)
- [Results: Bias between Ripe RIS v4,v6 and BGPtools v4,v6](#risv4v6-bgpv4v6-bias)

## Bias in Interent monitoring infrastructure including BGPtools
The results below are generated with the script `example_script_calculate_bias.py`, which calculates the bias of monitoring infrastructure along different dimentions, and (i) prints the results in the terminal, (ii) saves them in a csv file (folder `./data`) and (ii) generates radar plots for visualizing the bias.

#### Overview - Table with bias values 
The following table shows the bias of each set of monitors (columns) along different dimensions (rows)
```
                                      RIPE RIS (all)          RouteViews (all)        RIPE RIS + RouteViews (all)   bgptools (all)
### LOCATION INFO ###
RIR region                                   0.0727	           0.0052	              0.0310	                 0.1150
Location (country)	                     0.1985	           0.1807	              0.1353	                 0.3477
Location (continent)	                     0.0683	           0.0081	              0.0305	                 0.0995

### NETWORK SIZE INFO ### 
Customer cone (#ASNs)	                     0.1666	           0.1913	              0.1597	                 0.0542
Customer cone (#prefixes)	             0.1714	           0.2264	              0.1728	                 0.0355
Customer cone (#addresses)	             0.1884	           0.2033	              0.1793	                 0.0352
AS hegemony	                             0.1515	           0.2046	              0.1509	                 0.0263

### TOPOLOGY INFO ###
#neighbors (total)	                     0.3870	           0.3414	              0.3614	                 0.0885
#neighbors (peers)	                     0.3731	           0.3287	              0.3515	                 0.0839
#neighbors (customers)	                     0.1482	           0.1803	              0.1445                     0.0491
#neighbors (providers)	                     0.1418	           0.1443	              0.1373	                 0.0364
 
### IXP-RELATED INFO ###
#IXPs (PeeringDB)	                     0.1812	           0.1832	              0.1644	                 0.1059             
#facilities (PeeringDB)	                     0.1338	           0.1546	              0.1259	                 0.0288
Peering policy (PeeringDB)	             0.0138	           0.0197	              0.0126	                 0.0011

### NETWORK TYPE INFO ###
Network type (PeeringDB)	             0.1160	           0.1121	              0.1096	                 0.1167
Traffic ratio (PeeringDB)	             0.0940	           0.0880	              0.0843	                 0.0404
Traffic volume (PeeringDB)	             0.0378	           0.0774	              0.0386	                 0.0706  
Scope (PeeringDB)	                     0.1343	           0.1181	              0.1054	                 0.1362
Personal ASN	                             0.0065	           0.0022	              0.0053	                 0.10

```



#### Results: Bias between RIPE RIS & Routeviews & RIPE RIS + Routeviews & BGPtools

bias - radar plot
:-------------------------:
![Radar plot - ris_rv_ris+rv_bgp_bias](./figures/fig_radar_RIPERIS_RV_RIPERIS+RV_bgptools.png?raw=true) 

In this radar plot that compares RIPE RIS, RouteViews, RIPE RIS+RouteViews and bgptools, the first three suffer from bias in topology info dimension, especially #neighbours total and peers, while bgptools in this dimension have significantly less bias. The same happens in #IXPs (PeeringDB) dimension, however the bias is smaller proportionaly. 
On the other hand, bgptools suffer from bias in Location (country) dimension, more than all the other monitors. Then RIPE RIS monitors follow and after come the Routeviews ones. Also, we observe most bias in bgptools in Scope (PeeringDB) and finally Network type (PeeringDB), which all belong to Network type info dimension group. 

#### Results: Detailed distributions

Here, we present the detailed distributions per dimension (based on which the bias is calculated). The following figures depict the distributions of values for all ASes and for RIPE RIS, RouteViews and BGPtools monitors, along different dimensions. CDFs are used for numerical dimensions (e.g., number of neighbors), and histograms for categorical dimensions (e.g., type of network).

You can click on a figure to zoom in. The results below are generated with the same script (``example_script_calculate_bias.py``). All images can be found in the `./figures/` folder.  

**Location related dimensions**

&nbsp;|RIR region|Location (continent)|&nbsp;| &nbsp;
:---:|:---:|:---:|:---:|:---:
&nbsp; |![](./figures/Fig_Histogram_AS_rank_source.png?raw=true)| ![](./figures/Fig_Histogram_AS_rank_continent.png?raw=true)|&nbsp;|&nbsp;


**Network size dimensions**

Customer cone (#ASNs) | Customer cone (#prefixes) | Customer cone (#addresses) | AS hegemony | &nbsp;
:---:|:---:|:---:|:---:|:---:
![](./figures/Fig_CDF_AS_rank_numberAsns.png?raw=true)|![](./figures/Fig_CDF_AS_rank_numberPrefixes.png?raw=true)|![](./figures/Fig_CDF_AS_rank_numberAddresses.png?raw=true)|![](./figures/Fig_CDF_AS_hegemony.png?raw=true)|&nbsp;


**Topology related dimensions**

#neighbors (total)|#neighbors (peers)|#neighbors (customers)|#neighbors (providers)|&nbsp;
:---:|:---:|:---:|:---:|:---:
![](./figures/Fig_CDF_AS_rank_total.png?raw=true)|![](./figures/Fig_CDF_AS_rank_peer.png?raw=true)|![](./figures/Fig_CDF_AS_rank_customer.png?raw=true)|![](./figures/Fig_CDF_AS_rank_provider.png?raw=true)|&nbsp;


**IXP related dimensions**

&nbsp;|#IXPs (PeeringDB)|#facilities (PeeringDB)|Peering policy (PeeringDB)|&nbsp;
:---:|:---:|:---:|:---:|:---:
&nbsp;|![](./figures/Fig_CDF_peeringDB_ix_count.png?raw=true)|![](./figures/Fig_CDF_peeringDB_fac_count.png?raw=true)|![](./figures/Fig_Histogram_peeringDB_policy_general.png?raw=true)|&nbsp;


**Network type dimensions**

Network type (PeeringDB)|Traffic ratio (PeeringDB)|Traffic volume (PeeringDB)|Scope (PeeringDB)|Personal ASN
:---:|:---:|:---:|:---:|:---:
![](./figures/Fig_Histogram_peeringDB_info_type.png?raw=true)|![](./figures/Fig_Histogram_peeringDB_info_ratio.png?raw=true)|![](./figures/Fig_Histogram_peeringDB_info_traffic.png?raw=true)|![](./figures/Fig_Histogram_peeringDB_info_scope.png?raw=true)|![](./figures/Fig_Histogram_is_personal_AS.png?raw=true)


#### Bias in BGPtools detailed

bias - radar plot
:-------------------------:
![Radar plot - bias](./figures/fig_radar_bgptools.png?raw=true) 

In this radar plot, all bgptools monitors, only the IPv4, and only the IPv6 are compared in terms of bias in the different dimensions. It is obvserved that the Location (country) dimension has a lot of bias in all three cases, with the IPv4 monitors having the most. Then, we can see bias in IPv6 from RIR region, Personal ASN, to Scope DB, all in similar amount, with Scope DB including the IPv4 monitors too. Again, there is bias for all three cases in Network type (PeeringDB), in similar amount, in #IXPs (PeeringDB), with IPv4 monitors having the most, and the same goes between total number of neighbors and number of neighbours peers. Finally in all Customer cone dimension group we spot bias only in IPv4 monitors.

#### Bias between Ripe RIS IPv4,IPv6 and BGPtools IPv4, IPv6

bias - radar plot      
RIPE RIS IPv4 vs BGPtools IPv4 - bias            |  RIPE RIS IPv6 vs BGPtools IPv6 - bias  
:-------------------------:|:-------------------------:
![Radar plot - bias - bgptoolsv4_RIPERISv4](./figures/fig_radar_bgptoolsv4_RIPERISv4.png?raw=true)  |  ![Radar plot - bias - bgptoolsv6_RIPERISv6](./figures/fig_radar_bgptoolsv6_RIPERISv6.png?raw=true)

In the plots above, bgptools IPv4 monitors are compared to RIPE RIS IPv4 monitors, and bgptools IPv6 to RIPE RIS IPv6 ones.

In the figure for IPv4 monitors, we spot bias in the dimensions mentioned/seen in the last radar plot about all bgptools, having a lot of bias in Location (country), then Scope, Network type, #IXPs, between total number of neighbors and number of neighbours peers and a little in Customer cone. 
On the other hand, RIPE RIS monitors have much less bias in Location (country), the same amount in Scope, and Network type, and some in traffic ratio too. Then, there is bias between all IXP related, Topology related, and Customer cone bias groups, with the most bias spotted in (again) in total number of neighbors and number of neighbours peers, but the amount is much bigger.

In the figure for IPv6 monitors, the observations are very similar to IPv4, with less bias in Location for bgptools, more in RIR region and Personal ASN.  


