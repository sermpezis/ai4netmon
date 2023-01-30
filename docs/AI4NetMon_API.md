# The AI4NetMon API

We also provide an open API, which provides the bias scores for different IMPs of custom sets of vantage points. The API can be accessed at [https://ai4netmon.csd.auth.gr/api/](https://ai4netmon.csd.auth.gr/api/), and at this moment it provides the following endpoints (the API  is expected to be updated with extra functionality) 
- `/bias/{IMP}`
- `/asn/{ASN}`
whose use and functionality is described below.






### Bias scores `/bias/{IMP}` 
It takes as input either (i) the name of an IMP (see below for options) or (ii) a list of ASNs whose bias scores along all dimensions are returned.

Options for `IMP` parameter are:
- `Atlas`
- `RIS`
- `RouteViews`
- `RIS&RouteViews` (combined peers of both IMPs)
- `rrc00` (the peers of the RIPE RIS route collector with id 00)
- ...
- `rrc26`
- `?asn={ASN1}&asn={ASN2}&...&asn={ASNn}` (for a custom list of ASNs)

Example request:
[https://ai4netmon.csd.auth.gr/api/bias/RIS](https://ai4netmon.csd.auth.gr/api/bias/RIS)

Example output:
```yaml
{
    "RIR region": "0.0744",
    "Location (country)": "0.2193",
    "Location (continent)": "0.0709",
    "Customer cone (#ASNs)": "0.1805",
    "Customer cone (#prefixes)": "0.1857",
    "Customer cone (#addresses)": "0.2258",
    "AS hegemony": "0.1666",
    "#neighbors (total)": "0.46",
    "#neighbors (peers)": "0.4178",
    "#neighbors (customers)": "0.1494",
    "#neighbors (providers)": "0.1686",
    "#IXPs (PeeringDB)": "0.1822",
    "#facilities (PeeringDB)": "0.1362",
    "Peering policy (PeeringDB)": "0.0131",
    "Network type (PeeringDB)": "0.135",
    "Traffic ratio (PeeringDB)": "0.0951",
    "Traffic volume (PeeringDB)": "0.0421",
    "Scope (PeeringDB)": "0.1401",
    "Personal ASN": "0.0057"
}
````


Example request:
[https://ai4netmon.csd.auth.gr/api/bias/?asn=174&asn=1299&asn=2497&asn=3320&asn=3333](https://ai4netmon.csd.auth.gr/api/bias/?asn=174&asn=1299&asn=2497&asn=3320&asn=3333)

Example output:
```yaml
{
    "bias": {
        "RIR region": 0.12376846135988594,
        "Location (country)": 0.6574981157148462,
        "Location (continent)": 0.1475609152716093,
        "Customer cone (#ASNs)": 0.9452835754435239,
        "Customer cone (#prefixes)": 0.9142599397522675,
        "Customer cone (#addresses)": 0.8893599575719471,
        "AS hegemony": 0.7397630932662153,
        "#neighbors (total)": 0.991991170161001,
        "#neighbors (peers)": 0.9697221236565499,
        "#neighbors (customers)": 0.3590052963506119,
        "#neighbors (providers)": 0.2371842660443679,
        "#IXPs (PeeringDB)": 0.15783383510696097,
        "#facilities (PeeringDB)": 0.40724663740884676,
        "Peering policy (PeeringDB)": 0.7950874827804562,
        "Network type (PeeringDB)": 0.7830735184190359,
        "Traffic ratio (PeeringDB)": 0.4528003321949956,
        "Traffic volume (PeeringDB)": 0.7413179710952622,
        "Scope (PeeringDB)": 0.8645364212170968,
        "Personal ASN": 0.005261960347474111
    }, ,"#ASNs found":5,"#ASNs not found":0
 }
````

  



### ASN attributes `/asn/{ASN}` 
It takes the AS number (integer, e.g., `3333`) as parameter and returns all its attributes (for a detailed list, refer to the [doc](./Bias_in_IMPs.md) descrining our dataset)

Example request:
[https://ai4netmon.csd.auth.gr/api/asn/3333](https://ai4netmon.csd.auth.gr/api/asn/3333)

Example output:
```yaml
{
    "ASN": 3333,
    "AS_rank_rank": "5823.0",
    "AS_rank_source": "RIPE",
    "AS_rank_longitude": "4.90666750434161",
    "AS_rank_latitude": "52.3788892637641",
    "AS_rank_numberAsns": "3.0",
    "AS_rank_numberPrefixes": "48.0",
    "AS_rank_numberAddresses": "16416.0",
    "AS_rank_total": "320.0",
    "AS_rank_customer": "2.0",
    "AS_rank_peer": "315.0",
    "AS_rank_provider": "3.0",
    "AS_rank_iso": "NL",
    "AS_rank_continent": "Europe",
    "is_personal_AS": "",
    "peeringDB_info_ratio": "Balanced",
    "peeringDB_info_traffic": "1000.0",
    "peeringDB_info_scope": "Global",
    "peeringDB_info_type": "Non-Profit",
    "peeringDB_info_prefixes4": "30.0",
    "peeringDB_info_prefixes6": "20.0",
    "peeringDB_policy_general": "Selective",
    "peeringDB_ix_count": "1.0",
    "peeringDB_fac_count": "0.0",
    "peeringDB_created": "2012-11-09T06:06:08Z",
    "is_in_peeringDB": "1.0",
    "AS_hegemony": "2.17223250085813e-06",
    "nb_atlas_probes_v4": "7.0",
    "nb_atlas_probes_v6": "7.0",
    "nb_atlas_anchors": "2.0",
    "is_ris_peer_v4": "1",
    "is_ris_peer_v6": "",
    "is_routeviews_peer": "1.0",
    "AS_rel_degree": "319.0",
    "cti_top": "",
    "cti_origin": "0.00935128086572",
    "ASDB_C1L1": "Computer and Information Technology",
    "ASDB_C1L2": "Computer and Information Technology_Internet Service Provider (ISP)"
}
``` 
