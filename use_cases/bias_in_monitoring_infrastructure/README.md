# Use case: Quantifying the bias in the Internet monitoring infrastructure

The Internet monitoring infrastructure is not uniformly deployed in ASes. It is known that there are biases towards network sizes or types (e.g., large ISPs) or location (e.g., more monitors in Europe). The analysis and results aim to answer questions such as:
* How biased is the infrastructure? (e.g., from a scale from 0 to 1, where 0 is no bias)
* Is RIPE RIS more biased than RIPE Atlas? If yes, how much? What about RouteViews?
* Is the infrastructure more biased in terms of location or network sizes?


#### Table of Contents  
- [Bias metrics](#bias-metrics)  
- [Results: Bias in Interent monitoring](#results-bias-in-interent-monitoring-infrastructure)  
- [Results: Detailed distributions](#results-detailed-distributions)



## Bias metrics
To quantify the bias along a dimension (e.g., network type _peeringDB_info_type_), we compare the distribution of all ASes vs the distribution of the ASes that host monitors. For example, let the fraction of all ASes with specific types be _{large network: 0.2, medium network: 0.3, small network: 0.5}_ and the fraction of ASes with monitors be _{large network: 0.4, medium network: 0.4, small network: 0.2}_; then we need to quantify the difference among the two distributions represented by the vectors _[0.2, 0.3, 0.5]_ and _[0.4, 0.4, 0.2]_. To quantify the difference in the two distributions, we can use any of the following metrics 
- [**Kullback–Leibler (KL) divergence**]( https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence ) (in fact, we use a smoothed version to bound it to values up to 1); _unless otherwise specified, this is the default metric we use_
- [**Total variation (TV) distance**]( https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures ), which is the sum of the distances of two distributions P and Q (or, the L1 norm), i.e., _TV=0.5*sum{|P-Q|}_
- **Max variation distance**, which is the max distances among two distributions P and Q (or, the L1-inf norm), i.e., _max{|P-Q|}_

All metrics take values in the interval [0,1], where 0 corresponds to no bias, and larger values correspond to more bias. Each metric has a different interpretation, and results from one metric should not be compared with results from another metric in a quantitative way (only qualitative comparison).

## Results: Bias in Interent monitoring infrastructure 
The results below are generated with the script `example_script_calculate_bias.py`, which calculates the bias of monitoring infrastructure along different dimentions, and (i) prints the results in the terminal, (ii) saves them in a csv file (folder `./data`) and (ii) generates radar plots for visualizing the bias.

#### Overview - Table with bias values 
The following table shows the bias of each set of monitors (columns) along different dimensions (rows)
```
                                      RIPE RIS (all)    RIPE Atlas (all)  RouteViews (all)
### LOCATION INFO ###
RIR region                            0.06              0.06              0.00
Location (country)                    0.22              0.10              0.21
Location (continent)                  0.06              0.06              0.01

### NETWORK SIZE INFO ### 
Customer cone (#ASNs)                 0.22              0.07              0.23
Customer cone (#prefixes)             0.25              0.11              0.31
Customer cone (#addresses)            0.28              0.23              0.28
AS hegemony                           0.16              0.04              0.18

### TOPOLOGY INFO ###
#neighbors (total)                    0.57              0.12              0.44
#neighbors (peers)                    0.55              0.07              0.44
#neighbors (customers)                0.20              0.06              0.22
#neighbors (providers)                0.18              0.06              0.16

### IXP-RELATED INFO ###
#IXPs (PeeringDB)                     0.25              0.03              0.23
#facilities (PeeringDB)               0.20              0.03              0.18
Peering policy (PeeringDB)            0.03              0.01              0.02

### NETWORK TYPE INFO ###
Network type (PeeringDB)              0.15              0.03              0.13
Traffic ratio (PeeringDB)             0.12              0.02              0.09
Traffic volume (PeeringDB)            0.08              0.02              0.13
Scope (PeeringDB)                     0.16              0.04              0.17
Personal ASN                          0.00              0.00              0.00

```



#### Overview - Bias in RIPE infrastructure
A visual presentation of the bias is given below in the radar plot on the left (reminder: larger values, i.e., far from the center, corresponds to more bias); on the right, a more detailed version of the same plot, where the bias is depicted for IPv4 and IPv6 set of monitors


RIPE - bias - radar plot             |  RIPE - bias - radar plot (detailed)
:-------------------------:|:-------------------------:
![Radar plot - bias](./figures/fig_radar_RIPE.png?raw=true)  |  ![Radar plot - bias detailed](./figures/fig_radar_RIPE_detailed.png?raw=true)



Below the same plot with using as bias metrics the Total variation and Max variation distances

RIPE - bias (Total Variation)             |  RIPE - bias (Max Variation)
:-------------------------:|:-------------------------:
![Radar plot - bias - tv](./figures/fig_radar_RIPE_tv.png?raw=true)  |  ![Radar plot - bias - max](./figures/fig_radar_RIPE_max.png?raw=true)



#### Bias in RIPE RIS vs. RouteViews route collectors

Below we compare the bias between RIPE RIS and RouteViews route collectors

RIPE vs RouteViews - bias (KL)             |  RIPE vs RouteViews - bias (TV)
:-------------------------:|:-------------------------:
![Radar plot - bias - ripe rv](./figures/fig_radar_RIPE_RV.png?raw=true)  |  ![Radar plot - bias - ripe rv tv](./figures/fig_radar_RIPE_RV_tv.png?raw=true)





## Results: Detailed distributions

Here, we present the detailed distributions per dimension (based on which the bias is calculated). The following figures depict the distributions of values for all ASes and for RIPE NCC monitors, along different dimensions. CDFs are used for numerical dimensions (e.g., number of neighbors), and histograms for categorical dimensions (e.g., type of network).

You can click on a figure to zoom in. The results below are generated with the script `generate_distribution_plots.py`. All images can be found in the `./figures/` folder.  

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
