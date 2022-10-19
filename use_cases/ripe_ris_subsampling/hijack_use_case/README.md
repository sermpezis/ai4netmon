# Use case: Select a subset of RIPE RIS peers to improve the hijack impact estimation metric

### Introduction

In this README, we present the `hijack` use-case. The goal is to select a good subset of RIPE RIS peers, using unsupervised learning methods, in order to improve the hijack impact estimation metric. The main findings are in hijack.ipynb notebook, where the data are processed, clustering methods are tested to select subsets of monitors and extended analysis using plots is performed, to provide visual ways for interesting findings.
There are also some scripts, that use other ways of subsampling, such as greedy methods.

### Definitions and data

In a hijack event, two ASes (the legitimate AS and the hijacker AS) announce the same prefix and the rest of ASes route their traffic either to the legitimate or the hijacker AS. The impact of the AS is defined as the fraction of ASes that route their traffic to the hijacker AS. For example, if we have in total 1000 ASes, and 300 of them route to the hijacker, then the _actual_ hijack impact in 30%. The _estimated_ hijack impact is the corresponding percentage we can calculate from what the monitors see. In the previous example, if there are 50 monitors (e.g., RIPE RIS peers) and 25 of them route traffic to the hijacker AS, then the _estimated_ hijack impact is 50%. The difference between the actual and the esimated hijack impact is the error (in our example, the error is |30%-50%| = 20%).

The *basic* dataframe is loaded from **`./data/impact__CAIDA20190801_sims2000_hijackType0_per_monitor_onlyRC_NEW_with_mon_ASNs.csv`** file and it is used for calculation of the actual and estimated hijack impacts. It contains simulation results, where each simulation is a hijack event. The columns are:

- column 1:  ASN of the legitimate AS
- column 2:  ASN of the hijacker AS
- column 3:  total numer of ASes
- column 4:  do not take it into account
- column 5:  total numer of ASes that route to the hijacker
- column 6:  total numer of monitors
- column 7:  total numer of monitors that route to the hijacker
- columns 8-end: 0 if the corresponding monitor (whose ASN is given in the first header row) routes to the legitimate AS and 1 if it routes to the hijacker AS. (i.e., the sum of the columns 8-end in each row is equal to the number in column 5).

So, using the columns of the dataframe, the _actual_ hijack impact is calculated by column5/column3, while the _estimated_ hijack impact by column7/column6.

After calculating the impacts, all actual or estimated impact <0 or >1 are set to NaN. Then, the error (as mentioned above) is calculated.

### Goals and motivation

As said, the goal of the hijack impact estimation analysis is to improve the estimation error, by selecting good sets of monitors to estimate the error. A way that has been tested, to achieve this improvement, is by using supervised learning ([paper](https://arxiv.org/abs/2105.02346)), training an linear regression model on a set of previous observations. While the method mentioned proved to be optimal, we need to try unsupervised ways of subsampling, as they are more generic for other use cases (where there is no groundtruth data), but also can provide useful insights for deploying new monitors.  

The unsupervised approach contains the following steps:

* Check if there is correlation between monitors or network characteristics and errors. Also, check if correlations between important monitors (based on LRE weights) and errors. 
* Select subsets of monitors using clustering algorithms
* Select subsets of monitors based on their [bias](https://github.com/sermpezis/ai4netmon/tree/main/use_cases/bias_in_monitoring_infrastructure).

A method followed to be compared with the unsupervised approaches, was to select randomly X monitors (i.e., columns) from the *basic* dataframe and calculate the estimated impact from the values of these columns. Then, calculate the error from the actual impact. Plot RMSE(errors) vs X. For X = 10, 20, 30, 40, 50, 100, 150, 200, max number of monitors=300.

Below, the plot shows the average RMSE over 10 runs (y-axis) vs. the number of monitors (x-axis). We ran the method 10 times, because of the randomness in the selection of monitors.

![alt text](https://raw.githubusercontent.com/sermpezis/ai4netmon/main/use_cases/ripe_ris_subsampling/hijack_use_case/images/avgRMSEvsRandom.png)

### Problem characterization

In the first step of our approach mentionted before, the correlation between network characteristics of monitors (hijacker AS and legitimate AS) and error for every monitor is worth to be explored. 
We considered only topology and network size features of our [aggregated dataframe](https://raw.githubusercontent.com/sermpezis/ai4netmon/main/data/aggregate_data/asn_aggregate_data.csv).
So, combining the *basic* dataframe of hijack analysis and the *aggregated* one, we compile a new dataframe which contains the columns:

* ASN
* error
* features

while in the rows, first we have all the legitimate ASNs and then all the hijacker ASNs.

After the compilation, the pearson correlation is calculated between the network features and the errors. Although, we did not discover any, neither strong or weak, correlation between them, as well as no insights that coulb help in selecting monitors. The heatmaps of correlation matrices can be checked in the [notebook](https://github.com/sermpezis/ai4netmon/blob/main/use_cases/ripe_ris_subsampling/hijack_use_case/hijack.ipynb). 

Following the same approach, we calculate the pearson correlation for network characteristics and Linear regression models weights of the monitors, presented in the paper mentioned before. The findings once again show that there is not correlation between the above aspects. A table of those correlations is showed below, first for 50 and then all RIPE RIS + Routeviews (RC) monitors, and secondly 50 and then all Ripe Atlas (RA) monitors.


Features | AS rank numbers Asns| AS rank numbers prefixes | AS rank numbers addresses | AS hegemony | AS rank total | AS rank peers | AS rank costumers | AS rank provider | peeringDB ix count | peeringDB fac count | peeringDB info prefixes4 | peeringDB info prefixes6 | nb atlas probes v4 | nb atlas probes v6
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- 
RC50 Correlations w weights | 0.16 | 0.22 | 0.22 | -0.033 | -0.25 | 0.42 | -0.0096 | 0.095 | -0.44 | -0.35 | 0.093 | 0.048 | 0.15 | 0.29 
RC All Correlations w weights | 0.19 | 0.2 | 0.19 | 0.13 | -0.075 | -0.17 | 0.23 | 0.0055 | -0.04 | 0.061 | 0.11 | 0.07 | 0.16 | 0.12 
RA50 Correlations w weights | 0.32 | 0.34 | 0.21 | 0.35 | 0.041 | -0.065 | 0.33 | 0.078 | 0.16 | 0.59 | 0.23 | 0.2 | 0.46 | 0.33 
RA All Correlations w weights | 0.076 | 0.12 | 0.088 | 0.076 | -0.013 | -0.088 | 0.12 | 0.035 | 0.041 | 0.046 | 0.1 | 0.078 | 0.055 | 0.13 

### Subsampling with clustering

Regarding the clustering, we try to find optimal subsets of RC monitors. In the *basic* dataframe, there are 2000 hijack cases and 288 columns that correspond to monitors. The method we follow is to reverse the dataframe, and use the monitors as observations and use the columns as features. We try k-means to cluster the monitors based on the 2000 observations.

K-means algorithm is used with various numbes of clusters (5, 10, 20, 100). To extract the subsets, we select X monitors by iteratively selecting one monitor per cluster, and so on. E.g., if we have 5 clusters and we want to select X=20 monitors, select 4 monitors from each cluster.

After we have extracted the subsets, we select randomly X monitors and calculate the estimated impact and then the errors, in the same way we did with the *basic* dataframe. We calculate the RMSE for X = 10, 20, 30, 40, 50, 100, 150, 200, 288, and run the random selection for 50 times, to get the average RMSE.  Below, the plots of avg RMSE per 50 runs vs Number of monitors are shown, for every cluster.




