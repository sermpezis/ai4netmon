# Subsampling Internet Measurement Platforms (to Reduce Bias)


**Summary**: This article presents how one can select subsets ("subsampling") of the vantage points of the _Internet Measurement Platforms (IMPs)_, such as, RIPE Atlas, RIPE RIS, or RouteViews, in order to decrease their bias and get significantly more representative measurement data. More specifically, it provides (i) results for the bias of subsampling in IMPs, and (ii) a method for efficient subsampling.

For more information about the bias in IMPs and our methods, read first the [Bias in Internet Measurement Platforms](./Bias_in_IMPs) article.


#### Table of Contents  

- [Introduction](#introduction): Background and scope of the article 
- [Overall bias score](#overall-bias-score): Definition and options for the overall bias score (i.e., a single bias score value that aggregates the bias scores along all bias dimensions) 
- [Bias vs. number of vantage points](#bias-vs-number-of-vantage-points): Initial results that show how the bias of IMPs changes when considering only a (randomly selected) subset of their vantage points. Also, results related to:
	- Automatic probe selection in RIPE Atlas
	- Bias for each RIPE RIS route collector
- [Subsampling algorithm](#subsampling-algorithm): Algorithm for subsampling from IMPs aiming to decrease bias
- [Results](#results): Analysis and visualization of the results for bias reduction based on subsampling

## Introduction

The bias of the IMPs is due to the fact that some types of networks are under-represented and others over-represented (e.g., networks in Asia and Europe, respectively, for RIPE Atlas). To decrease the bias, we need to have a balanced representation of all network types. One way to achieve this is by using only a subset of the existing vantage points whose types are over-represented. 

_Remark:_ Another option for decreasing bias is by deploying extra vantage points to the under-represented network types. We analyze and discuss this case in another [article](./Extending_IMPs.md). 


## Overall bias score
As presented in the [Bias in Internet Measurement Platforms](./Bias_in_IMPs) article, a set of vantage points has a bias score value per dimension; for example, a set of vantage points may have 0.2 bias score in terms of location, and 0.6 bias score in terms of topology.

In this section, we first define and discuss "what" we aim to improve (i.e., the objective function). On one hand, one can aim to reduce the bias along one dimension (e.g., location), however, this may be very limited or even increase a lot the bias in other dimensions (e.g., topology). On the other hand, aiming to reduce the bias along many dimensions may lead to the following trade-off: what if a set of vantage points A reduces a lot the bias along a dimension and another set of vantage points B reduces a lot the bias along another dimenion? which one should be preferred? 

**Definition of the overall bias score:** We need to define an _overall bias score_ that takes into account the bias score along all dimensions. One obvious way to do this is to consider the _average_ bias score over all the dimensions of interest. 

> In the following results in this article, we consider the _average_ bias score

However, there can many other ways that may be reasonable for aggregating bias scores. Let us formalize how one can build a function for aggregating bias scores of different dimensions: 
- Consider a set of monitors `X`. 
- And consider some bias dimensions with id `1, 2, ..., N`. 
- Let us denote the bias score of set `X` along the dimension `i` as `BS_{i}`
- The _overall bias score_ of set `X` is given by function `f` of all bias scores, i.e., `BS = f(BS_{1}, BS_{2}, ..., BS_{N})` 


**Options for overall bias score:** Some plausible options for the aggregation function `f` are:
- _Average_: `BS = 1/N * sum_{i=1,2,..., N} BS_{i}` 
	- this metric considers all dimensions equally important
- _Weighted average_: `BS = sum_{i=1,2,..., N} w_{i}*BS_{i}` 
	- in this metric the importance of each dimension is determined by the value `w_{i}`
	- e.g., higher values `w_{i}` mean that the dimension `i` is more important
	- e.g., setting `w_{i}` equal to 0 means that we neglect this dimension
- _Maximum_: `BS = max_{i=1,2,..., N} BS_{i}`  
	- this is a "stricter" metric that considers the "worst case" among all dimensions
- _Balanced_:  `BS = 1 - prod_{i=1,2,..., N} (1-BS_{i})` 
	- a metric that is equal to 1 is at least one `BS_{i}` is 1, and is equal to 0 if _all_ `BS_{i}` are 0
	- it gets lower when the bias scores along all dimensions are relatively equal; e.g., if for a set A the bias scores are BS_{1}=0.5 and BS_{2}=0.5, and for another set B the bias scores are BS_{1}=0.2 and BS_{2}=0.8, then the overall bias scores given are BS(A)=0.75 < BS(B)=0.84


## Bias vs. number of vantage points 

We first present a preliminary analysis on how the bias score of an IMP changes when using only a subset of its vantage points. We perform subsampling randomly, i.e., without optimizing for reducing bias, but as a sensitivity analysis of bias vs. number of vantage points.

The following Figure shows the average bias for different sample sizes drawn randomly from either the entire population of ASes (`all`) or one of the three IMPs. Lines correspond to averages over 100 sampling iterations, and errorbars indicate 95% confidence intervals. For ease of comparison, dashed lines correspond to the bias values of using the entire infrastructure (i.e., all vantage points of the respective IMPs). 

![Random subsampling](../paper/figures/Fig_bias_vs_sampling_TOTAL.png?raw=true)
:-------------------------:
**Figure 1**: Bias vs. number of vantage points (random subsampling)



We observe that: 
- the bias decreases with the sample size (as expected)
- random subsampling from the entire population (`all`) has always lower bias than IMPs for the same number of vantage points
- even for very small sample sizes ( >20 vantage points), random sampling has lower bias than the _entire_ sets of RIPE RIS and RouteViews vantage points (see dashed lines), while the same holds for RIPE Atlas for >40 vantage points.

> Random subsampling from IMPs does _not_ decrease bias! Is it possible to decrease it with a more sophisticated subsampling method? (spoiler: yes!)


### Automatic probe selection in RIPE Atlas

RIPE Atlas users can either select specific probes to use in their measurements or not specify them (which is is the default choice; with parameters 10 probes from "worldwide locations"; [documentation](https://atlas.ripe.net/docs/udm/\#probe-selection)). In the latter case, RIPE Atlas has an automated algorithm to assign probes to a measurements, which prioritises probes with less load over more loaded probes, which makes the probe selection procedure not equivalent to true random sampling. 

In the Figure 2 below, we present how the RIPE Atlas selection algorithm, "Atlas (platform)", performs compared to random sampling from either all RIPE Atlas probes, "Atlas (random)"; and as a baseline a hypothetical scenario with random sampling from all ASes ("all"). We considered the sets of probes that the RIPE Atlas platform returned when we initiated measurements with parameters `type="area"``value="WW"`. Lines correspond to averages over 100 sampling iterations, and errorbars indicate 95% confidence intervals. 

We observe that _when using the RIPE Atlas algorithm for selecting probes, "Atlas (platform)", then the bias is significantly higher compared to selecting randomly probes, "Atlas (random)"_. In fact, the bias is almost two times higher. This indicates that even with the existing infrastructure, RIPE Atlas users could decrease bias by 50% by not depending on the built-in probe selection process, but select random probes themselves.

![](../paper/figures/Fig_bias_vs_sampling_real_Atlas_TOTAL.png?raw=true)|![](./figures/Fig_scatter_bias_vs_sampling_per_rrc.png?raw=true)
:---:|:---:
Bias in automatic probe selection in RIPE Atlas | Bias vs. number of peers in RIPE RIS route collectors
**Figure 2** | **Figure 3**


### Bias for each RIPE RIS route collector

Feeds from a single route collector may be used in cases that there are processing limitations (e.g., in terms of real-timeness or storage) due to the large volume of data. Figure 3 above presents the average bias score per RIPE RIS route collector (RRC) (i.e., the bias of the set of peers of the route collector) in relation to its number of peers. Overall, there is a clear (negative) correlation between the number of peers and the bias score of a route collector. Nevertheless, the size of a route collector does not predict its bias as (i) the three RRCs (rrc01, rrc03, rrc12) that are significantly larger (>80 members) than the rest of RRCs, are not less biased (in fact, there are several smaller RRCs with lower bias) and (ii) there are several medium-size RRCs (and even some with only 10-20 monitors) that have relatively low bias. The three multihop RCs (rrc00, rrc24, rrc25) are less biased than most of the non-multihop RRCs (which are deployed at IXPs).

**Online visualization tool**: We provide an online tool (ObservableHQ notebook) that presents interactive visualizations of the radar plots depicting the bias scores of individual RRCs along all the dimensions we considered. The tool is available at [https://observablehq.com/@pavlos/ai4netmon-bias-per-route-collector](https://observablehq.com/@pavlos/ai4netmon-bias-per-route-collector)



## Subsampling algorithm

In this section, we present two algorithms we designed to efficiently select subsets of vantage points (subsampling) of an IMP aiming to reduce its overall bias score, `BS` (as it is defines [above](#overall-bias-score)).

More specifically, if we denote the set of vantage points of an IMP as `VP`, then the goal is to find a set `S` with `k` vantage points, which has the minimum overall bias score. Solving optimally the problem is NP-hard, hence, we design two heuristic algorithms: (i) a _greedy_ algorithm and (ii) a _sorting_ algorithm. The _greedy_ is more efficient than the _sorting_, but of higher computational complexity (i.e., slower)

**The Greedy algorithm** starts by considering the entire set of vantage points, i.e., `S=VP`. Then, for every vantage point, it calculates the resulting bias score if we remove this vantage point from the IMP. And, it removes from the set `S` the vantage point that would decrease the most the bias score. It repeats the above process for the updated set `S` by removing one vantage point at each iteration, until the remaining set is of the desired size `k`. 

```
INPUT: vantage points VP, subset size k
RETURNS: subset S

S = VP 		// initialize the set S with all vantage points
WHILE {|S|>k} DO
	FOR {v in VP} DO
		W(v)  = BS(S-v)		// overall bias score of the set S without the vantage point v 
	END FOR
	v = argmin W 	// find the vantage point that reduces most the bias score ...
	S = S-v 		// ... and remove it from the set S
END WHILE
RETURN S
```

**The Sorting algorithm** initially calculates for every vantage point the resulting bias score if it is removed from the set `VP`, and then it removes (without further calculations) the `|VP|-k` vantage points that correspond to the lowest bias scores.

```
INPUT: vantage points VP, subset size k
RETURNS: subset S

S = VP 		// initialize the set S with all vantage points
FOR {v in VP} DO
	W(v)  = BS(S-v)		// overall bias score of the set S without the vantage point v 
END FOR
WHILE {|S|>k} DO
	v = argmin W 	// find the vantage point that reduces most the bias score ...
	S = S-v 		// ... and remove it from the set S
END WHILE
RETURN S
```




## Results

The following figures show the overall bias score (y-axis) for subsets of the IMPs selected by the _greedy_ algorithm (continuous lines) or the _sorting_ algorithm (dashed lines) for varying set sizes `k` (x-axis). The bias score values at the rightmost part of the curves correspond to the bias score of the entire set of vantage points `VP`. 

![](../paper/figures/fig_bias_vs_sampling_naive_and_greedy_only_RC.png?raw=true)|![](..paper/figures/fig_bias_vs_sampling_naive_and_greedy_only_Atlas.png?raw=true)
:---:|:---:
Subsampling vs. Bias score: RIPE RIS and RouteViews | Subsampling vs. Bias score: RIPE Atlas
**Figure 4** | **Figure 5**

Key findings:
- In all cases, the subsampling with the greedy algorithm has a similar behavior: the bias score decreases as we remove vantage points (i.e., moving to the left of the x-axis) to a minimum point, and then it increases again as the sub-set sizes become small. 

- The minumum point is achieved at samples sizes `k` that are one order of magnitude less than the size of the entire IMP (note the logarithmic scale of the x-axis).

- For large subset sizes the performance of both algorithms is similar. However, as the subset sizes decrease, the sorting algorithm performs worse, since it does not take into account in its decisions the combined effect of removing more than one vantage points

> RIPE RIS and RouteViews: the sweet spot of 50 peers.

- RIPE RIS and RouteViews have a similar behavior, with sample sizes of around 50 vantage points (i.e., peers) achieving the lowest bias score values of less than 0.04, which is _four times lower than the bias score of the entire IMPs_. Comparing the curves with the curve of random sampling of Figure 1, we can see that for sample sizes `k<100`, the subsampling algorithm can select sets that have lower bias than what a random sample (of the same size) from the entire population of networks has on average.

> RIPE Atlas: almost zero bias with a few hundreds of probes

- In the case of RIPE Atlas the bias score is less than 0.01 (i.e., six times lower than the entire set of all Atlas probes) for subsets of a few hundreds of vantage points (`84 < k < 976`). The minimum value is achieved for around 300 probes. In the Atlas case, the greedy algorithm performs better than random sampling for samples sizes `k<1000`.  