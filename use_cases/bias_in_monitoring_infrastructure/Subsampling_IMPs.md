# Subsampling Internet Measurement Platforms (to Reduce Bias)


**Summary**: This article presents how one can select subsets ("subsampling") of the vantage points of the _Internet Measurement Platforms (IMPs)_, such as, RIPE Atlas, RIPE RIS, or RouteViews, in order to decrease their bias and get significantly more representative measurement data. More specifically, it provides (i) results for the bias of subsampling in IMPs, and (ii) a method for efficient subsampling.

For more information about the bias in IMPs and our methods, read first the [Bias in Internet Measurement Platforms](./Bias_in_IMPs) article.


#### Table of Contents  

- [Introduction](#introduction): Background and scope of the article 


## Introduction

The bias of the IMPs is due to the fact that some types of networks are under-represented and others over-represented (e.g., networks in Asia and Europe, respectively, for RIPE Atlas). To decrease the bias, we need to have a balanced representation of all network types. One way to achieve this is by using only a subset of the existing vantage points whose types are over-represented. 

_Remark:_ Another option for decreasing bias is by deploying extra vantage points to the under-represented network types. We analyze and discuss this case in another [article](./Extending_IMPs.md). 


## Overall bias score
As presented in the [Bias in Internet Measurement Platforms](./Bias_in_IMPs) article, a set of vantage points has a bias score value per dimension; for example, a set of vantage points may have 0.2 bias score in terms of location, and 0.6 bias score in terms of topology.

In this section, we first define and discuss "what" we aim to improve (i.e., the objective function). On one hand, one can aim to reduce the bias along one dimension (e.g., location), however, this may be very limited or even increase a lot the bias in other dimensions (e.g., topology). On the other hand, aiming to reduce the bias along many dimensions may lead to the following trade-off: what if a set of vantage points A reduces a lot the bias along a dimension and another set of vantage points B reduces a lot the bias along another dimenion? which one should be preferred? 

**Definition of the overall bias score:** We need to define an _overall bias score_ that takes into account the bias score along all dimensions. One obvious way to do this is to consider the _average_ bias score over all the dimensions of interest. 

> In the folliwng results in this article, we consider the average bias score

However, there can many other ways that may be reasonable for aggregating bias scores. Let us formalize how one can build a function for aggregating bias scores of different dimensions: 
- Consider a set of monitors `X`. 
- And consider some bias dimensions with id `1, 2, ..., N`. 
- Let us denote the bias score of set `X` along the dimension `i` as `BS_{i}`
- The _overall bias score_ of set `X` is given by function `f` of all bias scores, i.e., `BS = f(BS_{1}, BS_{2}, ..., BS_{N})` 


**Options for overall bias score:** Some plausible options for the aggregation function `f` are:
- _Average_: `BS = 1/N * sum_{i=1,2,..., N} BS_{i}` (this considers all dimensions equally important)
- _Weighted average_: `BS = sum_{i=1,2,..., N} w_{i}*BS_{i}` (the importance of each dimension is determined by the value `w_{i}`; e.g., higher values `w_{i}` mean that the dimension `i` is more important; e.g., setting `w_{i}` equal to 0 means that we neglect this dimension)
- _Maximum_: `BS = max_{i=1,2,..., N} BS_{i}`  (this is a stricter metric that considers the "worst case" among all dimensions)


## Preliminary analysis: Bias vs. number of vantage points 
While the current set of vantage points is clearly not optimal in terms of bias, we wonder how biases changes when we only use a smaller random set of vantage points (e.g., measurements with few Atlas probes due to rate/credit limits, or collecting feeds from a subset of route collectors peers due to the large volumes of data). Figure~\ref{fig:bias-vs-sample-size-total} shows the average bias for different sample sizes drawn randomly from either the entire population of ASes (`all') or one of the three \imps. Lines correspond to averages over 100 sampling iterations, and errorbars indicate 95\% confidence intervals. For ease of comparison, dashed lines correspond to the bias values of using the entire infrastructure (i.e., the values in Table~\ref{table:bias-infra-vs-random}). We observe that: \one the bias decreases with the sample size (as expected), \two random sampling has always lower bias (for the same number of \vps), and \three even for very small sample sizes ($\geq$20 \vps), random sampling has lower bias than the \textit{entire} sets of RIPE RIS and \rv \vps (see dashed lines), while the same holds for RIPE Atlas for $\geq$40 \vps.
