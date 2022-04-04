# Bias in RIPE Atlas: The role of the automatic probe selection algorithm

The RIPE Atlas infrastructure (similarly to RIPE RIS and RouteViews) is not uniformly deployed in all types of networks and/or all around the world. As a result, measurements from RIPE Atlas inherently have some bias; see our RIPE Labs article on ["Bias in Internet Measurement Infrastructure"](https://labs.ripe.net/author/pavlos_sermpezis/bias-in-internet-measurement-infrastructure/) for a detailed presentation of this issue.

In this post, we focus on the issue of bias when conducting measurements with a few probes (from 10 up to 100), which is a common scenario of how network operators use the RIPE Atlas service:

- When using only a few probes, how biased are our measurements?
- Can we do better? and, if yes, how much?
- Does the RIPE Atlas algorithm for automatically selecting probes perform well in terms of bias? [spoiler: No!]

### Results
The results in the following figure provide initial answers to the above questions. The methodology for our experiments is detailed below, and then we provide a discussion of the main findings.

![](./figures/Fig_bias_vs_sampling_real_Atlas_TOTAL.png?raw=true)

*The figure shows the average bias for different sample sizes **drawn randomly** from the entire population of ASes ("all") or from the set of RIPE Atlas probes ("RIPE Atlas"). We also considered the sets of probes that the RIPE Atlas service returned when we initiated measurements with parameters type="area" and value="WW" ("RIPE Atlas real"). Lines correspond to averages over 100 sampling iterations, and errorbars indicate 95\% confidence intervals. For ease of comparison, the dashed line corresponds to the bias of the entire RIPE Atlas infrastructure.*


### Methodology
Experiments We did measurement experiments as follows:
- For each experiment, we selected a set of M probes (the number of probes corresponds to the **x-axis** of the plot). 
- We calculated the bias score of the selected set of probes (i.e., a score that captures how far the characteristics of the ASes that host those probes differ differ from the total population of ASes; the higher the score, the more the bias)
- For statistical robustness, we repeated the above process 100 times (each time we selected a different set of probes), and we calculated the average of the 100 bias scores; this is depicted in the **y-axis** of the plot.


We compare three different methods for selecting probes:
- **RIPE Atlas algorithm**: We initiated measurements using the RIPE Atlas API with measurement parameters _type="area"_ and _value="WW"_, and considered the set of probes that were automatically assigned by RIPE Atlas to each measurement.
- **RIPE Atlas (random)**: We randomly selected a set of ASes that host a RIPE Atlas probe.
- **All ASes (random)**: We randomly selected a set of ASes from the entire population of ASes (i.e., either they host probes or not); this corresponds to a hypothetical scenario, where all ASes host probes.


### Interpretation 
From the results in the above figure, we can do the following observations that also answer to the questions stated earlier: 
- The bias decreases with the sample size (as expected).
- For small number of probes(from M=10 to M=50), selecting _randomly_ RIPE Atlas probes ("RIPE Atlas") does not impose higher bias than selecting from any AS ("all"), which is the best one could do; i.e., there is no room for improvement.
- For larger number of probes (M>50), the bias due to RIPE Atlas non-uniform deployments starts appearing. For example, for M=100 probes,the bias of selecting _randomly_ RIPE Atlas probes ("RIPE Atlas") is almost twice than the best we could do ("all").
- ***When using the RIPE Atlas algorithm for selecting probes ("RIPE Atlas real"), then the bias is significantly higher than selecting randomly probes ("RIPE Atlas")***. It is almost two times higher. This indicates that even with the existing infrastructure, RIPE Atlas could decrease to 50% the bias by changing the automated probe selection algorithm. Moreover, if new deployments are made, the bias could be decreased even more.