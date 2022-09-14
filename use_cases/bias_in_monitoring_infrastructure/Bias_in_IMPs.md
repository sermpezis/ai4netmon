# Bias in Internet Measurement Platforms

**Summary**: This article defines and quantifies the bias in Internet Measurement Platforms (IMPs), such as, RIPE Atlas, RIPE RIS, or RouteViews. More specifically, it provides (i) a dataset and methods to quantify the bias, (ii) analysis results (data and visualizations), and (iii) guidelines and discussions for using and generalizing the methodology.

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
- We present the detailed methodology to calculate the bias based on the dataset ([Bias metrics](#bias-metrics)); we provide also guidelthe .

## Bias definition
blah blah ...

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
df = pd.read_csv(AGGREGATE_DATA_FNAME, header=0, index_col=0)
```

## Bias metrics