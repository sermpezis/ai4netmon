# Bias in Internet Measurement Platforms

**Summary**: This article defines and quantifies the bias in _Internet Measurement Platforms (IMPs)_, such as, RIPE Atlas, RIPE RIS, or RouteViews. More specifically, it provides (i) a dataset and methods to quantify the bias, (ii) analysis results (data and visualizations), and (iii) guidelines and discussions for using and generalizing the methodology.



#### Table of Contents  

- [Introduction](#introduction): Background and scope of the project 
- [Bias definition](#bias-definition): Introduction to the concept of bias
- [Data](#data): Presentation of the dataset, which contains multiple information (related to location, topology, etc.) about the networks in the Internet, and based on which the bias is calculated
- [Bias metrics](#bias-metrics): Definition of the metrics that are used to quantify bias, and the related methodology  
- [Results](#results): Analysis and visualization results for the bias in IMPs
- [Appendix](#appendix)
    - [A](#a:-bias-calculation-options): Bias calculation options
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

**_Examples of bias in IMPs_**: Some examples of biases (of different dimensions) in IMPs are the following:
- _location bias_: The percentage of all ASNs (population) that are located in Europe is around 30\%, whereas the percentage of RIPE Atlas probes (sample population) that are located in Europe is higher than 60\%. 
- _connectivity bias_: The average number of peering links for an ASN (population) is around 10, whereas the average number of peering links of ASNs that provide feed to RIPE RIS is more than 400. 
- _network type bias_: The percentage of all ASNs (population) that ar registered in PeeringDB as "Network Service Providers (NSPs)" is around 15%, whereas the corresponding percentage among the ASNs that provide feed to RouteViews is 40%.

We can see that bias is not the same along all dimensions, or accross all IMPs. In Section [Results](#results), we present detailed results and visualizations about the bias in the IMPs.


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

The bias (along a dimension/characteristic) is defined as a difference between two distributions: population vs. sample (see [Bias definition](#bias-definition)). For example, the population can be the entire set of ASes, a sample can be the ASes that peer with RIPE RIS, and the distributions can be the values of the number of peering links. In the dataframe, the aforementioned distributions could be retrieved as follows:
```python
dimension = "AS_rank_total"     # the dataframe column corresponding to the total number of neighbors of an AS
population_distribution = df[dimension].tolist()
sample_distribution = df.loc[df['is_ris_peer_v4']==1, dimension].tolist()
```

**_Identify bias_**: To identify if there is a statistically significant difference between the two distributions, one can run a statistical test. There are several statistical tests that could be applied. We select to use the widely used Kolmogorov-Smirnov (or, KS-test), which is a nonparametric test that compares two distributions (two-sample KS-test). This can be done as (documentation of the method [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html)):
```python
from scipy.stats import kstest
statistic, p_value = kstest(population_distribution, sample_distribution)
```
If the `p_value` is small (typically, less than 0.05), then the test rejects the null hypothesis that the two distributions are identical (and, thus, there exists bias in this dimension).


**_Quantify bias (the "bias score")_**: To quantify the bias along a dimension means to quantify the difference between the population and sample distributions. There exist several metrics to quantify the distance between two distributions. Among the most common is the [Kullback–Leibler (KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence), which is defined as follows:
- Let `P = [p(1), p(2), ..., p(K)]` be the population distribution, where `p(i)` denotes the probability that a sample of the population takes the i-th value. For example, in the example of Table 1, the distribution for countries would be `P = [0.7, 0.3]`.
- Let `Q = [q(1), q(2), ..., q(K)]` be the sample distribution; in the example of Table 1, the sample distribution for countries would be `Q = [0.8, 0.2]`.

Then, the KL-divergence (or, the **_bias score_, BS**) is calculated as
```
BS = sum_{i=1,...,K} p(i) * log(p(i)/q(i))
```

In our framework, we use a normalized (smoothed) version of the KL-divergence as the bias score for a dimension, which takes values between 0 (no bias) and 1 (very biased); the larger the bias score, the more biased the sample is. To make it easy to calculate the bias, we provide a method that takes care of the necessary preprocessing and calculations of the metrics, and which can be used, for example, as follows:
```python
from ai4netmon.Analysis.bias import bias_utils as bu
params={'data_type':'numerical', 'bins':10, 'alpha':0.01}
BS = bu.bias_score(population_distribution, sample_distribution, method='kl_divergence', params=**params)
```
We also provide examples of how to calculate the bias scores for all dimensions in the example scripts in [this folder](https://github.com/sermpezis/ai4netmon/tree/main/use_cases/bias_in_monitoring_infrastructure). For example, to calculate the bias score along all dimensions, one can simply do:
```python
from ai4netmon.Analysis.bias import bias_utils as bu
population_df = df
sample_df = df.loc[df['is_ris_peer_v4']==1, :]
sample_IDs = sample_df.index.tolist()
params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}
bias_df = bu.get_bias_of_monitor_set(df=population_df, monitor_list=sample_IDs, params=params)
```


For more options for calculating the bias score (e.g., different metrics or different populations) see [Appendix A](#a-bias-calculation-options)


## Appendix

### A: Bias calculation options

**Steps**
- [Optional] Select population; by default it can be the entire population, however, any custom population can be used as reference (see examples below). The selection of the population can be easily done by using the properties of the dataframe. Some examples can be:
    -`df_population = df` (the entire population of ASes)
    -`df_population = df[df["RIR region"]=="RIPE"]` (only ASes in the RIPE region)\
    -`df_population = df[df["Network type"]=="CDN"]` (only ASes that are registered as CDNs in the PeeringDB)
    -`df_population = df[df["Network type"]=="CDN"]` (only ASes that are registered as CDNs in the PeeringDB)
    -`df_population = df.loc[my_custom_list_of_ASNs, :]` (only ASes that belong to a custom list, e.g., `my_custom_list_of_ASNs = [174,1299,2497,...]`)
- Select set of interest
    -`df_sample = df.loc[df["is_ris_peer_v4"]==1,:]` (RIPE RIS peers - only IPv4)
    -`df_sample = df.loc[df["is_ris_peer_v4"]==1 | df["is_ris_peer_v6"]==1, :]` (all RIPE RIS peers)
    -`df_sample = df.loc[df["is_routeviews_peer"]==1, :]` (all RouteViews peers)
    -`df_sample = df.loc[df["nb_atlas_probes_v4"]>1 | df["nb_atlas_probes_v6"]>1]` (all ASes with at least one IPv4/IPv6 RIPE Atlas probe)
