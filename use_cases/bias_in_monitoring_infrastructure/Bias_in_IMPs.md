# Bias in Internet Measurement Platforms

**Summary**: This article defines and quantifies the bias in _Internet Measurement Platforms (IMPs)_, such as, RIPE Atlas, RIPE RIS, or RouteViews. More specifically, it provides (i) a dataset and methods to quantify the bias, (ii) analysis results (data and visualizations), and (iii) guidelines and discussions for using and generalizing the methodology.



#### Table of Contents  

- [Introduction](#introduction): Background and scope of the project 
- [Bias definition](#bias-definition): Introduction to the concept of bias
- [Data](#data): Presentation of the dataset, which contains multiple information (related to location, topology, etc.) about the networks in the Internet, and based on which the bias is calculated
- [Bias metrics](#bias-metrics): Definition of the metrics that are used to quantify bias, and the related methodology  
- [Results](#results): Analysis and visualization results for the bias in IMPs
- [Appendix](#appendix)
    - [A](#a): blah blah
    - [B](#b): blah blah



## Introduction

Network operators and researchers frequently use Internet measurement platforms (IMPs), such as RIPE Atlas, RIPE RIS, or RouteViews for, e.g., monitoring network performance, detecting routing events, topology discovery, or route optimization. IMPs operate a wide array of globally distributed vantage points. RIPE Atlas hosts ~11000 measurements probes in ~3300 autonomous systems (ASes), RIPE RIS and RouteViews collect routing information from ~300 and ~500 ASes, respectively. Even though the number of vantage points is large, visibility in the ~70000 globally routed ASes is still partial. While this incompleteness aspect has been extensively studied, it is still unclear how _representative_ (wrt. the entire Internet) the view we have through the IMPs. Do we have equal visibility to all types of networks? And, if not, how biased our views are? To interpret the results of its measurements, users must understand a platform's limitations and biases. 

To this end, this article presents a framework to analyze the multi-dimensional (e.g., across location, topology, network types, etc.) biases of IMPs, as well as a detailed analysis of IMP biases:
- We first discuss what is bias, in general, and in the context of IMPS, in particular. ([Bias definition](#bias-definition))
- We then present a dataset we compiled with multiple information about ASes ([Data](#data)), based on which one can calculate the bias of the existing IMPs or of any other custom set of vantage points / networks. The dataset itself aggregates information from various online sources, and thus can be useful for many applications (other than bias).
- We present the detailed methodology to calculate the bias based on the dataset ([Bias metrics](#bias-metrics)); we also provide guidelines on how the methodology can be adapted according to the needs of the user.


## Bias definition

Let’s first try to understand the notion and implications of bias through a general example (see Table 1), and then proceed to the more complicated case of bias in IMPs.

|TABLE. 1|Men | Women | Country A | Country B|
|:---|:---:|:---:|:---:|:---:|
|Entire population|50%|50%|70%|30%|
|Survey sample    |80%|20%|80%|20%|

**_Defining bias_**: Assume a population of 100 people, of which 50 are men and 50 are women. If we do a survey and ask 10 of them, 8 men and 2 women, then the sample we get is biased towards men. Why? Because the fractions of men and women in the population are 50% and 50%, respectively, whereas in our sample they are 80% and 20%. In other words, we say that our sample is biased as there is a _difference in the distributions between the entire population and our sample_. 


**_Quantifying bias_**: Hence, to (i) _identify_ if there is bias in a sample and (ii) _quantify_ how much bias a sample has, one needs to calculate if there is statistical difference between the population and the sample distributions, and then measure the distribution distance among the two distributions. For both objectives there are standard statistical tests and measures; we present below in the [Bias metrics](#bias-metrics) section the metrics we use.


**_Bias dimensions_**: In the above example (Table 1), assume that in our population 70% of people are from country A and 30% from country B, and in our sample the corresponding fractions are 80% and 20%. Therefore, _in a dataset there may be multiple dimensions of bias_ (here, gender and country bias). In our example, gender bias is higher than country bias; i.e., _the extent of bias can be different among different dimensions_. In section [Data](#data), we present the dataset we compiled, which contains data for each network (and IMPs) along different dimensions.


**_Is bias a problem (for measurements)?_**: Let us consider that the goal of our survey focuses on the height of the population, i.e., we ask the survey participants what their height is, to calculate an average for the entire population. Then, probably we get biased results, since men, who are typically taller than women, are over-represented in our sample. Now, let us consider that our survey further focuses on the native language of individuals. For this second case, the gender-bias in our sample would not affect our findings. In contrast, the country-bias (e.g., see the right side of the Table 1) of our sample, may play a major role. In other words, _different bias dimensions (e.g., gender or country) may affect our measurements findings differently, depending on how they relates to the insights we want to gain_. In section **TBD** we demonstrate how a user can focus on only some dimensions of bias in our dataset and analysis. 


## Data

Each network or vantage point can be characterized by a multitude of features, such as, location, connectivity, traffic levels, etc.. We collect data from multiple online (public) data sources to compile a dataset, which contains multiple information for each AS (we use an AS-level granularity due to the availability of data; however, our methodology is applicable to more fine-grained levels, e.g., per vantage point or BGP prefix, given that data at this granularity are provided).

**Data sources**: We collect data from the following publicly available datasets:
- CAIDA AS-rank [link](https://asrank.caida.org/)
- CAIDA AS-relationship [link](https://publicdata.caida.org/datasets/as-relationships/) 
- PeeringDB (provided by CAIDA) [link](https://publicdata.caida.org/datasets/peeringdb/)
- RIPE Atlas probes [link](https://atlas.ripe.net/api/v2/probes)
- RIPE RIS route collectors [link](https://stat.ripe.net/data/ris-peers/data.json)
- RouteViews route collectors [link](http://www.routeviews.org/peers/peering-status.html)
- AS hegemony (by Internet Health Report) [link](https://ihr.iijlab.net/ihr/hegemony/)
- Country-level Transit Influence (CTI) [link](https://github.com/CAIDA/mapkit-cti-code)
- personal-use ASNs (by bgp.tools) [link](https://bgp.tools/tags/perso.txt)
- ASDB [link](https://asdb.stanford.edu)

**Dataset**: We compile and preprocess the information collected from the above data sources to a dataset that contains _**37 characteristics (columns)**_ about _**more than 100,000 ASNs (rows)**_. 

A subset of characteristics that are included in the dataset and are more relevant to the definition of bias are the following (grouped in categories):
- _Location-related_: RIR region; Country; Continent
- _Network size-related_: Customer cone (\#ASNs); Customer cone (\#prefixes); Customer cone (\#addresses); AS hegemony
- _Topology-related_: \#neighbors (total); \#neighbors (peers); \#neighbors (customers); \#neighbors (providers)
- _Interconnection/IXP-related_: \#IXPs; \#facilities; Peering policy
- _Network type-related_: Network type; Traffic ratio; Traffic volume; Scope; Personal ASN


The following image depicts the format of the dataset.


![Dataframe AI4NetMon](./figures/fig_ai4netmon_dataframe_example.png?raw=true)
:-------------------------:
Figure: Example of the compiled dataset

The dataset is available at the Github repository of the AI4NetMon project as a .csv file: [link](https://github.com/sermpezis/ai4netmon/tree/main/data/aggregate_data). 

It can be easily loaded in python as follows:
```python
import pandas as pd
URL = "https://raw.githubusercontent.com/sermpezis/ai4netmon/main/data/aggregate_data/asn_aggregate_data.csv"
df = pd.read_csv(URL, header=0, index_col=0)    # "index_col=0" sets the ASN as the index of the dataframe
```

## Bias metrics


To _identify_ this bias, one could run statistical tests (e.g.,  Kolmogorov-Smirnov test) to compare the two distributions. To further \textit{quantify} the bias, it is common to measure the \textit{distribution distance} among the population and the sample distributions (e.g., common metrics are the Kullback-Leibler divergence or the Total Variation distance). 




To quantify the bias along a dimension (e.g., network type _peeringDB_info_type_), we compare the distribution of all ASes vs the distribution of the ASes that host monitors. For example, let the fraction of all ASes with specific types be _{large network: 0.2, medium network: 0.3, small network: 0.5}_ and the fraction of ASes with monitors be _{large network: 0.4, medium network: 0.4, small network: 0.2}_; then we need to quantify the difference among the two distributions represented by the vectors _[0.2, 0.3, 0.5]_ and _[0.4, 0.4, 0.2]_. To quantify the difference in the two distributions, we can use any of the following metrics 
- [**Kullback–Leibler (KL) divergence**]( https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence ) (in fact, we use a smoothed version to bound it to values up to 1); _unless otherwise specified, this is the default metric we use_
- [**Total variation (TV) distance**]( https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures ), which is the sum of the distances of two distributions P and Q (or, the L1 norm), i.e., _TV=0.5*sum{|P-Q|}_
- **Max variation distance**, which is the max distances among two distributions P and Q (or, the L1-inf norm), i.e., _max{|P-Q|}_

All metrics take values in the interval [0,1], where 0 corresponds to no bias, and larger values correspond to more bias. Each metric has a different interpretation, and results from one metric should not be compared with results from another metric in a quantitative way (only qualitative comparison).


**Steps**
- [Optional] Select population; by default it can be the entire population, however, any custom population can be used as reference (see examples below). The selection of the population can be easily done by using the properties of the dataframe.
    - e.g., `df_population = df` (the entire population of ASes)
    - e.g., `df_population = df[df["RIR region"]=="RIPE"]` (only ASes in the RIPE region)\
    - e.g., `df_population = df[df["Network type"]=="CDN"]` (only ASes that are registered as CDNs in the PeeringDB)
    - e.g., `df_population = df[df["Network type"]=="CDN"]` (only ASes that are registered as CDNs in the PeeringDB)
    - e.g., `df_population = df.loc[my_custom_list_of_ASNs, :]` (only ASes that belong to a custom list, e.g., `my_custom_list_of_ASNs = [174,1299,2497,...]`)
- Select set of interest
    - e.g., `df_ris_peers = df[df["is_ris_peer"]==1]`
