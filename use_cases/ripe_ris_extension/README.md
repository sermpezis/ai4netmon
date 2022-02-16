## Bias by adding an extra ASN to the set of RIPE RIS monitors/peers

We calculate what is the value of bias when adding an extra ASN to the set of RIPE RIS monitors. We first calculate the bias for the set of RIPE RIS monitors. Then for every ASN, we calculate the bias of the set {RIPE RIS}U{ASN}. If the resulting bias is smaller, then the added ASN reduces the bias. 

We sort ASNs by the resulting bias in ascending order, i.e., the first ASN in the ordered list is the ASN that reduces the most the bias.

From the analysis, we **_omitted the stub ASes_**


#### Data 

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

Based on a preliminary analysis (see figures below) the most meaningful dimensions (i.e., keys of the dict in the json) to be further analyzed for this purpose are:
- total
- AS_rank_iso (i.e., location country)
- peeringDB_info_traffic



#### Analysis - CDFs of bias of {RIPE RIS}U{ASN} 

We calculate the distribution of the resulting bias values for the set {RIPE RIS}U{ASN} over all ASNs, i.e., the values [Bias({RIPE RIS}U{ASN1}), {RIPE RIS}U{ASN2}, {RIPE RIS}U{ASN3}, ...]. We plot the CDFs (blue lines) along the the bias that the RIPE RIS set has (red lines).



AS hegemony | Continent | #customers | Country
:---:|:---:|:---:|:---:
![](./figures/cdf_bias_diff__no_stubs_NEW_AS_hegemony.png?raw=true)|![](./figures/cdf_bias_diff__no_stubs_NEW_AS_rank_continent.png?raw=true)|![](./figures/cdf_bias_diff__no_stubs_NEW_AS_rank_customer.png?raw=true)|![](./figures/cdf_bias_diff__no_stubs_NEW_AS_rank_iso.png?raw=true)

customer cone (#addresses) | customer cone (#ASNs) | customer cone (#prefixes) | #peers
:---:|:---:|:---:|:---:
![](./figures/cdf_bias_diff__no_stubs_NEW_AS_rank_numberAddresses.png?raw=true)|![](./figures/cdf_bias_diff__no_stubs_NEW_AS_rank_numberAsns.png?raw=true)|![](./figures/cdf_bias_diff__no_stubs_NEW_AS_rank_numberPrefixes.png?raw=true)|![](./figures/cdf_bias_diff__no_stubs_NEW_AS_rank_peer.png?raw=true)

nb providers | RIR | #neighbors total | is personal AS
:---:|:---:|:---:|:---:
![](./figures/cdf_bias_diff__no_stubs_NEW_AS_rank_provider.png?raw=true)|![](./figures/cdf_bias_diff__no_stubs_NEW_AS_rank_source.png?raw=true)|![](./figures/cdf_bias_diff__no_stubs_NEW_AS_rank_total.png?raw=true)|![](./figures/cdf_bias_diff__no_stubs_NEW_is_personal_AS.png?raw=true)

nb facilities | traffic ratio | Scope | Traffic
:---:|:---:|:---:|:---:
![](./figures/cdf_bias_diff__no_stubs_NEW_peeringDB_fac_count.png?raw=true)|![](./figures/cdf_bias_diff__no_stubs_NEW_peeringDB_info_ratio.png?raw=true)|![](./figures/cdf_bias_diff__no_stubs_NEW_peeringDB_info_scope.png?raw=true)|![](./figures/cdf_bias_diff__no_stubs_NEW_peeringDB_info_traffic.png?raw=true)

Network type | #IXPs | peering policy | **Total Bias**
:---:|:---:|:---:|:---:
![](./figures/cdf_bias_diff__no_stubs_NEW_peeringDB_info_type.png?raw=true)|![](./figures/cdf_bias_diff__no_stubs_NEW_peeringDB_ix_count.png?raw=true)|![](./figures/cdf_bias_diff__no_stubs_NEW_peeringDB_policy_general.png?raw=true)|![](./figures/cdf_bias_diff__no_stubs_NEW_total.png?raw=true)