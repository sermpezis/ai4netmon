In practive when using the RIPE Atlas infrastructure, operators use only a subset of the RIPE Atlas probes for their measurements. The bias of a subset of probes is not the same with the bias when considering the entire RIPE Atlas infrastructure. The bias of a random "sample" or subset of probes is expected to have (on average) higher bias than the entire set of probes. 

This is clearly seen in the following radar plot, where we considered subsets / random samples of 10, 20, and 100 RIPE Atlas probes, and calculated their bias (for each case, we did 100 iterations and calculated the average values; e.g., we considered 100 sets of 10 RIPE Atals probes, calculated the Location bias for each of them, and the value presented in the plot for "RIPE Atlas (10 samples)" is the average over the 100 values).

We can see that the bias is significantly large when selecting only 10 probes. It decreases when considering more probes. With 100 probes the bias is quite close to the entire RIPE Atlas infrastructure (except for the location dimension), which indicates that with 100 probes we can achieve relatively good bias.


![](./figures/Fig_radar_RIPE_Atlas_sampling.png?raw=true)


In the figure below, we present the average bias (y-axis) for subsets of the Internet monitoring infrastucture of different sizes (x-axis). We also see here that the bias for subsets of the RIPE Atlas gets close to the value of the entire set of probes (dashed line) when considering at least a few hundreds of probes. 

![](./figures/Fig_bias_vs_sampling_TOTAL.png?raw=true)