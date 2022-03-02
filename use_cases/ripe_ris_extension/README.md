# Bias by adding an extra ASN to the set of RIPE RIS monitors/peers

We calculate what is the value of bias when adding an extra ASN to the set of RIPE RIS monitors. We first calculate the bias for the set of RIPE RIS monitors. Then for every ASN, we calculate the bias of the set {RIPE RIS}U{ASN}. If the resulting bias is smaller, then the added ASN reduces the bias. 

We sort ASNs by the resulting bias in ascending order, i.e., the first ASN in the ordered list is the ASN that reduces the most the bias.

From the analysis, we **_omitted the stub ASes_**




#### Table of Contents  
- [Data](#data)  
- [Results: CDFs of bias of RIPE RIS plus extra ASN](#results-cdfs-of-bias-of-ripe-ris-plus-extra-asn)  
- [Results: CDFs of characteristics of top-K extra ASNs that reduce most the bias](#results-cdfs-of-characteristics-of-top-k-extra-asns-that-reduce-most-the-bias)
- [Results: CDFs of characteristics of top-K extra ASNs that reduce most the bias (filtered by improvement)](#results-cdfs-of-characteristics-of-top-k-extra-asns-that-reduce-most-the-bias-filtered)


## Data 

The file `sorted_asns_by_ascending_biases.json` has the format 
```
{ 
  "total": [ASN1, ASN2, ...]
  "bias_dimension1": [ASN1, ASN2, ...], 
  "bias_dimension2": [ASN1, ASN2, ...],
  ...
}
```  
where the keys are the bias dimensions and the values are the ordered lists (the first AS in the list reduces the most the bias); the key `total` corresponds to the aggregated bias along all dimensions.

Based on a preliminary analysis (see results below) the most meaningful dimensions (i.e., keys of the dict in the json) to be further analyzed for this purpose are:
- total
- AS_rank_iso (i.e., location country)
- peeringDB_info_traffic



## Results: CDFs of bias of RIPE RIS plus extra ASN 

We calculate the distribution of the resulting bias values for the set {RIPE RIS}U{ASN} over all ASNs, i.e., the values [Bias({RIPE RIS}U{ASN1}), {RIPE RIS}U{ASN2}, {RIPE RIS}U{ASN3}, ...]. We plot the CDFs (blue lines) along the the bias that the RIPE RIS set has (red lines). See detailed plots [here](./Plots_characteristics_RIPE_RIS_plus_one.md)

## Results: CDFs of characteristics of top-K extra ASNs that reduce most the bias

We calculate the detailed distributions per dimension for the top-K (K=50 or K=200) extra monitors that could be added to RIPE RIS to decrease the bias the most. See detailed plots [here](./Plots_charactertistics_extra_monitors.md)

## Results: CDFs of characteristics of top-K extra ASNs that reduce most the bias filtered

We calculate the detailed distributions per dimension for the top-K (K=50 or K=200) extra monitors that could be added to RIPE RIS to decrease the bias the most; _we consider only extra monitors that bring proximity improvement more than 1000_. See detailed plots [here](./Plots_characteristics_extra_monitors_filtered.md)