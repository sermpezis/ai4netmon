#### Table of Contents  
- [Results: Bias in Interent monitoring](#results-bias-in-interent-monitoring-infrastructure)  
- [Results: Detailed distributions](#results-detailed-distributions)


## Results: Bias in Interent monitoring infrastructure including BGPtools
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

RIPE RIS vs RouteViews - bias (KL)             |  RIPE RIS vs RouteViews - bias (TV)
:-------------------------:|:-------------------------:
![Radar plot - bias - ripe rv](./figures/fig_radar_RIPE_RV.png?raw=true)  |  ![Radar plot - bias - ripe rv tv](./figures/fig_radar_RIPE_RV_tv.png?raw=true)



#### Bias in RIPE RIS special cases: (i) full feeders, (ii) extending infrastructure

Below we first compare the bias in RIPE RIS when considering all monitors and only full-feeders. We see that _**full feeders are considerably _more biased**_. 

Moreover, we present the expected bias of the RIPE RIS infrastructure if a number of N extra monitors are added as peers. We consider N = 10, 50, 100, 200 extra monitors, selected from an ordered list of ASNs that [we calculated]( https://github.com/sermpezis/ai4netmon/tree/dev/TEMP_pavlos/bias_sort_nonRIS_asns ) that would decrease most the bias (i.e., for N=50 we select the first 50 ASNs appearing in the list). The more monitors we add the less the bias (as expected), however, the figure allows to quantify the added gains.

RIPE RIS: all vs full-feeders             |  Extending RIPE RIS with extra monitors
:-------------------------:|:-------------------------:
![Radar plot - bias - ripe rv](./figures/fig_radar_RIPE_full.png?raw=true)  |  ![Radar plot - bias - ripe rv tv](./figures/fig_radar_RIPE_plus.png?raw=true)





## Results: Detailed distributions

Here, we present the detailed distributions per dimension (based on which the bias is calculated). The following figures depict the distributions of values for all ASes and for RIPE NCC monitors, along different dimensions. CDFs are used for numerical dimensions (e.g., number of neighbors), and histograms for categorical dimensions (e.g., type of network).

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
