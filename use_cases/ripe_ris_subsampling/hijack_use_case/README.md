### Use case: Select a subset of RIPE RIS peers to improve the hijack impact estimation metric

In a hijack event, two ASes (the legitimate AS and the hijacker AS) announce the same prefix and the rest of ASes route their traffic either to the legitimate or the hijacker AS. The impact of the AS is defined as the fraction of ASes that route their traffic to the hijacker AS. For example, if we have in total 1000 ASes, and 300 of them route to the hijacker, then the _actual_ hijack impact in 30%. The _estimated_ hijack impact is the corresponding percentage we can calculate from what the monitors see. In the previous example, if there are 50 monitors (e.g., RIPE RIS peers) and 25 of them route traffic to the hijacker AS, then the _estimated_ hijack impact is 50%. The difference between the actual and the esimated hijack impact is the error (in our example, the error is |30%-50%| = 20%).


Files:
* **`./data/impact__CAIDA20190801_sims2000_hijackType0_per_monitor_onlyRC_NEW_with_mon_ASNs.csv`**: File that contains simulation results of `nb_sims` simulations, where each simulation corresponds to a hijack event. The first row is the header; each of the other rows corresponds to a hijack event. The columns are:

- column 1:  ASN of the legitimate AS
- column 2:  ASN of the hijacker AS
- column 3:  total numer of ASes
- column 4:  do not take it into account
- column 5:  total numer of ASes that route to the hijacker
- column 6:  total numer of monitors
- column 7:  total numer of monitors that route to the hijacker
- columns 8-end: 0 if the corresponding monitor (whose ASN is given in the first header row) routes to the legitimate AS and 1 if it routes to the hijacker AS. (i.e., the sum of the columns 8-end in each row is equal to the number in column 5).

I.e., the actual impact is calculated as column5/column3, and the estimated as column7/column6 (set to NaN all values with actual or estimated impact <0 or >1). 


### Introduction

In this README, we present the `hijack` use-case. The goal is to select a good subset of RIPE RIS peers, using unsupervised learning methods, in order to improve the hijack impact estimation metric. The main findings are in hijack.ipynb notebook, where the data are processed, clustering methods are tested to select subsets of monitors and extended analysis using plots is performed, to provide visual ways for interesting findings.

### Definitions and data

In a hijack event, two ASes (the legitimate AS and the hijacker AS) announce the same prefix and the rest of ASes route their traffic either to the legitimate or the hijacker AS. The impact of the AS is defined as the fraction of ASes that route their traffic to the hijacker AS. For example, if we have in total 1000 ASes, and 300 of them route to the hijacker, then the _actual_ hijack impact in 30%. The _estimated_ hijack impact is the corresponding percentage we can calculate from what the monitors see. In the previous example, if there are 50 monitors (e.g., RIPE RIS peers) and 25 of them route traffic to the hijacker AS, then the _estimated_ hijack impact is 50%. The difference between the actual and the esimated hijack impact is the error (in our example, the error is |30%-50%| = 20%).

The dataframe is loaded from * **`./data/impact__CAIDA20190801_sims2000_hijackType0_per_monitor_onlyRC_NEW_with_mon_ASNs.csv`** file and it is used for calculation of the actual and estimated hijack impacts. It contains simulation results, where each simulation is a hijack event. The columns are:

- column 1:  ASN of the legitimate AS
- column 2:  ASN of the hijacker AS
- column 3:  total numer of ASes
- column 4:  do not take it into account
- column 5:  total numer of ASes that route to the hijacker
- column 6:  total numer of monitors
- column 7:  total numer of monitors that route to the hijacker
- columns 8-end: 0 if the corresponding monitor (whose ASN is given in the first header row) routes to the legitimate AS and 1 if it routes to the hijacker AS. (i.e., the sum of the columns 8-end in each row is equal to the number in column 5).

So, using the columns of the dataframe, the _actual_ hijack impact is calculated by column5/column3, while the _estimated_ hijack impact by column7/column6.

After calculating the impacts, all actual or estimated impact <0 or >1 are set to NaN.





