### Use case: Quantifying the bias in the Internet monitoring infrastructure

The Internet monitoring infrastructure is not uniformly deployed in ASes. It is known that there are biases towards network sizes or types (e.g., large ISPs) or location (e.g., more monitors in Europe). The analysis and results aim to answer questions such as:
* How biased is the infrastructure? (e.g., from a scale from 0 to 1, where 0 is no bias)
* Is RIPE RIS more biased than RIPE Atlas? If yes, how much?
* Is the infrastructure more biased in terms of location or network sizes?


To quantify the bias along a dimension (e.g., network type _peeringDB_info_type_), we compare the distribution of all ASes vs the distribution of the ASes that host monitors. For example, let the fraction of all ASes with specific types be _{large network: 0.2, medium network: 0.3, small network: 0.5}_ and the fraction of ASes with monitors be _{large network: 0.4, medium network: 0.4, small network: 0.2}_; then we need to quantify the difference among the two distributions represented by the vectors _[0.2, 0.3, 0.5]_ and _[0.4, 0.4, 0.2]_. We use the [Kullback–Leibler (KL) divergence]( https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence ) metric to quantify the difference in the two distributions (in fact, we use a smoothed version to bound it to values up to 1). The metric takes values in the interval [0,1], where 0 corresponds to no bias, and larger values correspond to more bias.


The results below are generated with the script `example_script_calculate_bias.py`, which calculates the bias of monitoring infrastructure along different dimentions, and (i) prints the results in the terminal and (ii) generates radar plots for visualizing the bias



The following table shows the bias of each set of monitors (columns) along different dimensions (rows)
```
DIMENSION                           RIPE RIS (all)    RIPE Atlas (all)
AS_rank_source                      0.06              0.06
AS_rank_iso                         0.22              0.10
AS_rank_continent                   0.06              0.06
is_personal_AS                      0.00              0.00
peeringDB_info_ratio                0.12              0.02
peeringDB_info_traffic              0.08              0.02
peeringDB_info_scope                0.16              0.04
peeringDB_info_type                 0.15              0.03
peeringDB_policy_general            0.03              0.01
AS_rank_numberAsns                  0.22              0.07
AS_rank_numberPrefixes              0.25              0.11
AS_rank_numberAddresses             0.28              0.23
AS_rank_total                       0.57              0.12
AS_rank_customer                    0.20              0.06
AS_rank_peer                        0.55              0.07
AS_rank_provider                    0.18              0.06
peeringDB_info_prefixes4            0.19              0.03
peeringDB_info_prefixes6            0.22              0.03
peeringDB_ix_count                  0.25              0.03
peeringDB_fac_count                 0.20              0.03
AS_hegemony                         0.16              0.04
```

A visual presentation of the bias is given below in the radar plot (reminder: larger values, i.e., far from the center, corresponds to more bias)

![Radar plot - bias](./fig_radar.png?raw=true)


A more detailed version of the above plot, where the bias is depicted for IPv4 and IPv6 set of monitors, is shown below

![Radar plot - bias detailed](./fig_radar_detailed.png?raw=true)