# Bias in Internet Measurement Platforms

**Summary**: This article defines and quantifies the bias in _Internet Measurement Platforms (IMPs)_, such as, RIPE Atlas, RIPE RIS, or RouteViews. More specifically, it provides (i) a dataset and methods to quantify the bias, (ii) analysis results (data and visualizations), and (iii) guidelines and discussions for using and generalizing the methodology.



#### Table of Contents  

- [Introduction](#introduction): Background and scope of the project 
- [Bias definition](#bias-definition): Introduction to the concept of bias
- [Data](#data): Presentation of the dataset, which contains multiple information (related to location, topology, etc.) about the networks in the Internet, and based on which the bias is calculated
- [Bias metrics](#bias-metrics): Definition of the metrics that are used to quantify bias, and the related methodology  
- [Results](#results): Analysis and visualization results for the bias in IMPs
	- [IMPs: overall bias characterization](#imps-overall-bias-characterization)
	- [Bias characterization: a more detailed view](#bias-characterization-a-more-detailed-view)
- [Appendix](#appendix)
    - [A](#a-bias-calculation-options): Bias calculation options for generalization and parametrization of the methodology and results
    - [B](#b-dataset-eda): Exploratory data analysis (EDA) of the dataset



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


**_Is bias a problem (for measurements)?_**: Let us consider that the goal of our survey focuses on the height of the population, i.e., we ask the survey participants what their height is, to calculate an average for the entire population. Then, probably we get biased results, since men, who are typically taller than women, are over-represented in our sample. Now, let us consider that our survey further focuses on the native language of individuals. For this second case, the gender-bias in our sample would not affect our findings. In contrast, the country-bias (e.g., see the right side of the Table 1) of our sample, may play a major role. In other words, _different bias dimensions (e.g., gender or country) may affect our measurements findings differently, depending on how they relates to the insights we want to gain_. 

A user can focus on only some dimensions of bias, based on the use case under investigation. For this reason, in our dataset and analysis we calculate the bias values for every dimension individually; then the user can select if they will use all these values or some of them, and/or how to aggregate them (see more details in Sections [Bias metrics](#bias-metrics), [Results](#results)), and [Appendix A](#a-bias-calculation-options)). 


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
**Figure 1**: Example of the compiled dataset

The dataset is available at the Github repository of the AI4NetMon project as a .csv file: [link](https://github.com/sermpezis/ai4netmon/tree/main/data/aggregate_data). 

It can be easily loaded in python as follows:
```python
import pandas as pd
URL = "https://raw.githubusercontent.com/sermpezis/ai4netmon/main/data/aggregate_data/asn_aggregate_data.csv"
df = pd.read_csv(URL, header=0, index_col=0)    # "index_col=0" sets the ASN as the index of the dataframe
```



## Bias metrics

The bias (along a dimension/characteristic) is defined as a difference between two distributions: population vs. sample (see [Bias definition](#bias-definition)). For example, the _population_ can be the entire set of ASes, a _sample_ can be the ASes that peer with RIPE RIS (in IPv4), and the _distributions_ can be the values of the number of peering links. In the dataframe, the aforementioned distributions could be retrieved as follows:
```python
dimension = "AS_rank_total"     # the dataframe column corresponding to the total number of neighbors of an AS
population_distribution = df[dimension].dropna().tolist()
sample_distribution = df.loc[df['is_ris_peer_v4']==1, dimension].dropna().tolist()
```

**_Identify bias_**: To identify if there is a statistically significant difference between the two distributions, one can run a statistical test. There are several statistical tests that could be applied. We select to use the widely used Kolmogorov-Smirnov (or, KS-test), which is a nonparametric test that compares two distributions (two-sample KS-test). This can be done as (documentation of the method [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html)):
```python
from scipy.stats import kstest
statistic, p_value = kstest(population_distribution, sample_distribution)
```
If the `p_value` is small (typically, less than 0.05), then the test rejects the null hypothesis that the two distributions are identical (and, thus, there exists bias in this dimension).


**_Quantify bias (the "bias score")_**: To quantify the bias along a dimension means to quantify the difference between the population and sample distributions. There exist several metrics to quantify the distance between two distributions. Among the most common is the [Kullback–Leibler (KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) (for other metrics see [Appendix A](#a-bias-calculation-options)), which is defined as follows:
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

The result of the above code (the `BS` variable; the bias score) is a single value, which is the KL-divergence. For example, at the time of writing this document the resuls is ```0.026```.


We also provide examples of how to calculate the bias scores for all dimensions in the example scripts (`example_script_*.py`) in [this folder](https://github.com/sermpezis/ai4netmon/tree/main/use_cases/bias_in_monitoring_infrastructure). For example, to calculate the bias score of the set of IPv4 RIPE RIS peers along all dimensions, one can simply do:
```python
from ai4netmon.Analysis.bias import bias_utils as bu
population_df = df
sample_df = df.loc[df['is_ris_peer_v4']==1, :]
sample_IDs = sample_df.index.tolist()
params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}
bias_df = bu.get_bias_of_monitor_set(df=population_df, monitor_list=sample_IDs, params=params)
```

The result of the above code is a dataframe as follows:
```
AS_rank_source  0.076378
AS_rank_iso     0.225580
AS_rank_continent   0.072400
AS_rank_numberAsns  0.190840
AS_rank_numberPrefixes  0.202917
AS_rank_numberAddresses     0.265934
AS_hegemony     0.168170
AS_rank_total   0.541757
AS_rank_peer    0.471208
AS_rank_customer    0.157465
AS_rank_provider    0.190659
peeringDB_ix_count  0.181456
peeringDB_fac_count     0.144553
peeringDB_policy_general    0.013354
peeringDB_info_type     0.136842
peeringDB_info_ratio    0.097839
peeringDB_info_traffic  0.044955
peeringDB_info_scope    0.138789
is_personal_AS  0.000000
```

For more options for calculating the bias score (e.g., different metrics or different populations) see [Appendix A](#a-bias-calculation-options)





## Results


In this section, we use our framework (dataset, bias metrics, code) to study the biases in the IMPs, by calculating and visualizing the bias scores along all dimensions.



### IMPs: overall bias characterization

> Is there bias? Definitely, yes!

We applied the KS-test for all platforms and dimensions. In almost all cases, the KS-test rejected the null hypothesis that the IMPs vantage points follow the same distribution as the entire population of ASes. 


**The bias radar plot:** The following figure shows a radar plot with bias scores for all dimensions. The colored lines (and their included area) correspond to the bias metric of a given IMP along a given dimension, e.g., the bias score for RIPE RIS (orange line) in the dimension “Location (country)” is 0.2. Larger bias scores (i.e., farther from the center) correspond to more bias,  e.g., in the dimension “Location (country)” RIPE RIS is more biased than RIPE Atlas (blue line). Values closer to the center indicate lower bias. 


![Bias radar plot all IMP](../paper/figures/fig_radar_all.png?raw=true)
:-------------------------:
**Figure 2**: Radar plot depicting the bias score for RIPE Atlas, RIPE RIS, and RouteViews over the different dimensions.

The table below shows the detailed bias score values of Figure 2.
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


Some key observations, are:

- While the bias of IMPs differs significantly by dimension, RIPE Atlas is substantially less biased than RIPE RIS and RouteViews along most dimensions.
- RIPE RIS has a significant topological bias (e.g., number of total neighbors or peers) as most of its collectors are deployed IXPs, where ASes establish many (peering) connections. While RouteViews has also significant bias in this dimension, RIPE Atlas has substantially lower bias. 
- RouteViews and RIPE RIS are also quite biased in terms of network size (see “Customer cone” dimensions), which is due to the fact that many large ISPs provide feeds to the route collectors. While having feeds from large ISPs may be desired in terms of visibility or coverage, users should be aware of it since it may lead to biased measurements.
- In most IXP-related and network type dimensions (that correspond to data mainly from PeeringDB), all platforms have relatively low bias; with an exception of RIPE RIS and RouteViews that are biased in terms of number of IXPs/facilities the monitors are connected to.
- There are small differences between RIPE RIS and RouteViews. RIPE RIS is more biased in terms of topology (number of neighbors, total and peers), whereas RouteViews is more biased in terms of network sizes (“Customer cone” and “AS hegemony” dimensions). In most IXP-related dimensions, both platforms have similar biases.
    

### Bias characterization: a more detailed view

Beyond the above basic analysis, in the following we present three plots that help to deepen our understanding of different IMP aspects. 


![](../paper/figures/fig_radar_only_RCs.png?raw=true)|![](../paper/figures/fig_radar_only_RCs_full_feeders.png?raw=true)|![](../paper/figures/fig_radar_Atlas_v4_v6.png?raw=true)
:---:|:---:|:---:
Combinign RIS & RouteViews | Full vs. all feeds |IPv4 vs IPv6 Atlas probes


**_Combining RIS and RouteViews_**: Using data from both RIPE RIS and RouteViews is common; hence, we analyze the combined bias. When considering vantage points from both projects, the bias slightly decreases in most dimensions. Interestingly, there are some exceptions, e.g., number of neighbors (total and peers), where it would be preferable (in terms of bias) to use only feeds from RouteViews.

**_Full vs. all feeds_**: Only around 300 peers of RIPE RIS and RouteViews provide feeds for the entire routing table ("full feeds"). We compare the bias of only feed peers with that of the entire platform separately for RIPE RIS and RouteViews. For RIPE RIS the increase in bias is small, whereas for RouteViews the set of full feeds is significantly more biased. In fact, while RIPE RIS is on average more biased than RouteViews, the opposite becomes true when considering only full feeds.


**_IPv4 vs IPv6 vantage points_**: We compare the set of ASes hosting IPv4, IPv6, and any RIPE Atlas probes. The set of networks hosting IPv6 probes is slightly more biased than networks hosting IPv4 probes in most dimensions. 






# Appendix

## A: Bias calculation options

When looking for bias in a dataset there may be several parameters that may depend on the use case; namely, what is the reference population (e.g., all ASes or only ASes of a specific characteristic), what is the sample population (e.g., RIPE RIS peers or RIPE Atlas probes), what dimensions of bias are of interest, etc. 

In this section, we provide some examples on how the users can use our code to parametrize their investigation of bias.

**_Population_**: by default the population is the entire population, however, any custom population can be used as reference (see examples below). The selection of the population can be easily done by using the properties of the dataframe. Some examples can be:
- `df_population = df` (the entire population of ASes)
- `df_population = df[df["RIR region"]=="RIPE"]` (only ASes in the RIPE region)
- `df_population = df[df["Network type"]=="CDN"]` (only ASes that are registered as CDNs in the PeeringDB)
- `df_population = df[df["Network type"]=="CDN"]` (only ASes that are registered as CDNs in the PeeringDB)
- `df_population = df.loc[my_custom_list_of_ASNs, :]` (only ASes that belong to a custom list, e.g., `my_custom_list_of_ASNs = [174,1299,2497,...]`)


**_Sample population_**: The sample population is typically the set of vantage points selected. These sets can be the sets of vantage points in the IMPs or any other custom set of vantage points (e.g., private measurement system) or even an hypothetical set of vantage points (whose bias is to be investigated; cf. the [doc](./Extending_IMPs.md) about extending IMPs and the [doc](./Subsampling_IMPs.md) about subsampling IMPs). Some examples can be:
- `df_sample = df.loc[df["is_ris_peer_v4"]==1,:]` (RIPE RIS peers - only IPv4)
- `df_sample = df.loc[df["is_ris_peer_v4"]==1 | df["is_ris_peer_v6"]==1, :]` (all RIPE RIS peers)
- `df_sample = df.loc[df["is_routeviews_peer"]==1, :]` (all RouteViews peers)
- `df_sample = df.loc[df["is_ris_peer_v4"]==1 | df["is_ris_peer_v6"]==1 | df["is_routeviews_peer"]==1, :]` (all RIPE RIS and RouteViews peers)
- `df_sample = df.loc[df["nb_atlas_probes_v4"]>1 | df["nb_atlas_probes_v6"]>1]` (all ASes with at least one IPv4/IPv6 RIPE Atlas probe)
- `df_sample = df.loc[my_custom_list_of_ASNs, :]` (only ASes that belong to a custom list, e.g., `my_custom_list_of_ASNs = [174,1299,2497,...]`)


**_Bias dimensions_**: Not all bias dimensions may be relevant for a use case (as discussed in the [Bias definition](#bias-definition) section). Users can select the bias dimension to investigate, as we presented in the [Bias metrics](#bias-metrics) section, e.g., to select only the dimension "total number of neighbors" one can do:
```python
dimension = "AS_rank_total"     # the dataframe column corresponding to the total number of neighbors of an AS
population_distribution = df_population[dimension].dropna().tolist()
sample_distribution = df_sample[dimension].dropna().tolist()
from ai4netmon.Analysis.bias import bias_utils as bu
params={'data_type':'numerical', 'bins':10, 'alpha':0.01}
BS = bu.bias_score(population_distribution, sample_distribution, method='kl_divergence', params=**params)
```
or calculate the bias score values for all dimensions (as we presented in the [Bias metrics](#bias-metrics) section), and just keep the values for the dimensions of interest
```python
from ai4netmon.Analysis.bias import bias_utils as bu
sample_IDs = df_sample.index.tolist()
bias_df = bu.get_bias_of_monitor_set(df=df_population, monitor_list=sample_IDs, params=params)
dimensions_of_interest = ["AS_rank_total", "AS_rank_numberAsns"] 	# total nb of neighbors and customer cone
print(bias_df.loc[dimensions_of_interest])
```

**_Bias metrics_**

As discussed in the [Bias metrics](#bias-metrics) section, to quantify the bias along a dimension, we compare the population distribution (`P`) vs the sample distribution (`Q`). To quantify the difference in the two distributions, there are several metrics. The following are the most common (and relevant to our purposes). Each metric returns a single value, which is the _bias score_. 

- [**Kullback–Leibler (KL) divergence**]( https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence ) (in fact, we use a smoothed version to bound it to values up to 1)
`BS = sum_{i=1,...,K} p(i) * log(p(i)/q(i))`

- [**Total variation (TV) distance**]( https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures ), which is the sum of the distances of two distributions P and Q (or, the L1 norm)
`BS = 0.5 * sum_{i=1,...,K} |p(i) - q(i)|`

- **Max variation distance**, which is the max distances among two distributions P and Q (or, the L-inf norm)
`BS = max_{i=1,...,K} p(i) |p(i) - q(i)|`


The main difference between KL-divergence and TV distance metrics, is that the former is more sensitive to changes in characteristics of lower probabilities `p(i)`. For example, let `P = [0.6,0.2,0.2]` and two distributions `QA = [0.7,0.1,0.2]` and `QB = [0.6,0.1,0.3]` that differ by `+/- 0.1` compared to `P`. While for the total variation (TV) it holds that `BS_TV(P,QA) = BS_TV(P,QB)`, for the KL-divergence it holds `BS_TV(P,QA) = BS_TV(P,QB)`, because the `+0.1` was at a characteristic with a lower probability in `QB`.

The main difference between the Max distance and the other metrics, is that the former accounts for the "worst case" (i.e., the maximum deviation between two distributions), whereas the latter measure the bias by averaging distances over the entire distribution. 

Some examples for selecting each metric in our code are the following:
```python
# KL-divergence
params={'method':'kl_divergence', 'bins':10, 'alpha':0.01}

# TV-distance
params={'method':'total_variation', 'bins':10}

# Max variation distance
params={'method':'max_variation', 'bins':10}
```
where the `alpha` parameter is used for the smoothed version of KL-divergence, and the `bins` parameter is independent of the bias metric (it is used to preprocess data); we recommend these values for both these parameters.


 In the following Figure we present the radar plot depicting the bias for the three bias metrics. While the actual values differ for different metrics, the qualitative findings (e.g., which infrastructure set is more biased) remain the same for the majority of dimensions.

![](../paper/figures/fig_radar_all.png?raw=true)|![](../paper/figures/fig_radar_all_tv.png?raw=true)|![](../paper/figures/fig_radar_all_max.png?raw=true)
:---:|:---:|:---:
KL-divergence | Total variation (TV) | Max distance





## B: Dataset EDA

To explore the data, some following commands are the following:

- Print the all the features (columns) of the dataset and their respective data types (numerical, categorical, etc.)
```python
df.dtypes
```

- Print the number of non-nan values (`count`) and range (min/max), percentiles, mean, and standard deviation of the values for the numerical columns.
```python
df.describe()
```

- To further facilitate the EDA, we provide a method that visualizes the distributions for all features (generates CDF plots for numerical features, e.g., number of neighbors, and histograms for categorical features, e.g., type of network), which can be called as simple as follows:
```python
from ai4netmon.Analysis.bias import generate_distribution_plots as gdp
gdp.plot_all(network_sets, FILENAME, save_json=False, show_plot=False)

```
Check this [example script](https://github.com/sermpezis/ai4netmon/blob/main/use_cases/bias_in_monitoring_infrastructure/example_script_calculate_bias.py) for how to define the `network_sets` (i.e., dataframes of the sets to be visualized) and `FILENAME` options (i.e., names of files to be saved)

The above code will produce the following figures (you can click on any figure to zoom in:

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

