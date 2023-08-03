# The AI4NetMon API

We also provide an open API, which provides the bias scores for different IMPs of custom sets of vantage points. The API can be accessed at [https://ai4netmon.csd.auth.gr/api/](https://ai4netmon.csd.auth.gr/api/), and at this moment it provides the following endpoints (the API  is expected to be updated with extra functionality) 
- `/bias/{imp}`
- `/bias/asn`
- `/bias/probe`
- `/bias/random/{imp}/{nb}`

- `/bias/cause/{imp}`
- `/bias/cause/asn`
- `/bias/cause/probe`

- `/asn/{ASN}`
- `/probe/{probe}`
- `/rrc/{rrc}`

- `/distributions/custom/asn`
- `/distributions/custom/probe`

- `/distributions/{imp}`

whose use and functionality is described below.



### Bias scores `/bias/{IMP}` 
It takes as input either (i) the name of an IMP (see below for options) or (ii) a list of ASNs (or probes ids), whose bias scores along all dimensions are returned. Also, there is the `/bias/random/{n}` endpoint, that takes as input a number n, and selects 10 sets of n random monitors, and returns their average bias.  

Options for `IMP` parameter are:
- `Atlas`
- `RIS`
- `RouteViews`
- `RIS&RouteViews` (combined peers of both IMPs)
- `rrc00` (the peers of the RIPE RIS route collector with id 00)
- ...
- `rrc26`
- `?asn={ASN1}&asn={ASN2}&...&asn={ASNn}` (for a custom list of ASNs)
- `?probe={PROBE1}&probe={PROBE2}&...&probe={PROBEn}&v4=false&v6=true`
- `/random/{imp}/{nb}`

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
[https://ai4netmon.csd.auth.gr/api/bias/asn/?asn=174&asn=1299&asn=2497&asn=3320&asn=3333](https://ai4netmon.csd.auth.gr/api/bias/asn/?asn=174&asn=1299&asn=2497&asn=3320&asn=3333)
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
Example request:
[https://ai4netmon.csd.auth.gr/api/bias/probe/?probe=1&probe=2&probe=5&probe=20&v4=true&v6=false](https://ai4netmon.csd.auth.gr/api/bias/probe/?probe=1&probe=2&probe=5&probe=20&v4=true&v6=false)

Example output:
```yaml
{
  "bias": {
    "RIR region": 0.5799627667076053,
    "Location (country)": 0.9726752156473064,
    "Location (continent)": 0.618554898834941,
    "Customer cone (#ASNs)": 0.929189036141135,
    "Customer cone (#prefixes)": 0.8932468728505935,
    "Customer cone (#addresses)": 0.9734242091991357,
    "AS hegemony": 0.9846431088746982,
    "Country influence (CTI origin)": 0.9614155629619617,
    "Country influence (CTI top)": 0.7324612892396218,
    "#neighbors (total)": 0.9747092620088461,
    "#neighbors (peers)": 0.9811585658359434,
    "#neighbors (customers)": 0.9545233087316963,
    "#neighbors (providers)": 0.8443285140440955,
    "#IXPs (PeeringDB)": 0.1760430595659005,
    "#facilities (PeeringDB)": 0.20043584893005417,
    "Peering policy (PeeringDB)": 0.8010782139760931,
    "Network type (PeeringDB)": 0.5253646670109279,
    "Traffic ratio (PeeringDB)": 0.617804217453592,
    "Traffic volume (PeeringDB)": 0.8067404058999731,
    "Scope (PeeringDB)": 0.5423224722152219,
    "Personal ASN": 0.005261960347474111,
    "ASDB C1L1": 0.40127916969426436,
    "ASDB C1L2": 0.41189629167264524
  },
  "#probes found": 2,
  "#probes not found": 2,
  "Not found probes": [
    1,
    20
  ]
}
````

Example request:
[https://ai4netmon.csd.auth.gr/api/bias/random/Atlas/50](https://ai4netmon.csd.auth.gr/api/bias/random/Atlas/50)

Example output:
```yaml
{
	"Atlas": {
		"50": {
			"RIR region": 0.0960812725502687,
			"Location (country)": 0.42670954396298405,
			"Location (continent)": 0.09488375013120072,
			"Customer cone (#ASNs)": 0.06326158344862856,
			"Customer cone (#prefixes)": 0.09656960203956257,
			"Customer cone (#addresses)": 0.27979872208309187,
			"AS hegemony": 0.10438643627257208,
			"Country influence (CTI origin)": 0.11585951044388224,
			"Country influence (CTI top)": 0.4179960644984281,
			"#neighbors (total)": 0.11877663790128326,
			"#neighbors (peers)": 0.07656724568388812,
			"#neighbors (customers)": 0.051589686551915204,
			"#neighbors (providers)": 0.08007021886582988,
			"#IXP (PeeringDB)": 0.055519699038940704,
			"#facilities (PeeringDB)": 0.05438067706692451,
			"Peering policy (PeeringDB)": 0.021132588970552275,
			"ASDB C1L1": 0.1857909358424337,
			"ASDB C1L2": 0.281695937531723,
			"Network type (PeeringDB)": 0.0652678730856838,
			"Traffic ratio (PeeringDB)": 0.0399300951324701,
			"Traffic volume (PeeringDB)": 0.1435172194927757,
			"Scope (PeeringDB)": 0.10980484065596188,
			"Personal ASN": 0.00304337008890343
		}
	}
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
### Probe attributes `/probe/{PROBEid}` 
It takes the probe id (integer, e.g., `5`) as parameter and returns all the attributes of the corresponding AS. There is a parameter (for a detailed list, refer to the [doc](./Bias_in_IMPs.md) descrining our dataset)

Example request:
[https://ai4netmon.csd.auth.gr/api/probe/5?v4=true&v6=true](https://ai4netmon.csd.auth.gr/api/probe/5?v4=true&v6=true)

Example output:
```yaml
{
      "ASN": "3265",
      "AS_rank_rank": "4369.0",
      "AS_rank_source": "RIPE",
      "AS_rank_longitude": "5.03625111963776",
      "AS_rank_latitude": "52.2149207016351",
      "AS_rank_numberAsns": "4.0",
      "AS_rank_numberPrefixes": "27.0",
      "AS_rank_numberAddresses": "1115392.0",
      "AS_rank_total": "51.0",
      "AS_rank_customer": "3.0",
      "AS_rank_peer": "45.0",
      "AS_rank_provider": "3.0",
      "AS_rank_iso": "NL",
      "AS_rank_continent": "Europe",
      "is_personal_AS": "",
      "peeringDB_info_ratio": "Mostly Outbound",
      "peeringDB_info_traffic": "10000.0",
      "peeringDB_info_scope": "Europe",
      "peeringDB_info_type": "Not Disclosed",
      "peeringDB_info_prefixes4": "42.0",
      "peeringDB_info_prefixes6": "15.0",
      "peeringDB_policy_general": "No",
      "peeringDB_ix_count": "0.0",
      "peeringDB_fac_count": "0.0",
      "peeringDB_created": "2004-07-28T00:00:00Z",
      "is_in_peeringDB": "1.0",
      "AS_hegemony": "0.0003635365752679",
      "nb_atlas_probes_v4": "3.0",
      "nb_atlas_probes_v6": "1.0",
      "nb_atlas_anchors": "1.0",
      "is_ris_peer_v4": "",
      "is_ris_peer_v6": "",
      "is_routeviews_peer": "",
      "AS_rel_degree": "5.0",
      "cti_top": "",
      "cti_origin": "2.26664657873",
      "ASDB_C1L1": "Computer and Information Technology",
      "ASDB_C1L2": "Computer and Information Technology_Internet Service Provider (ISP)"
}

``` 
### Route collectors attributes `/rrc/{rrc}` 
It takes the rrc (string, e.g., `rrc10`) as parameter and returns all the attributes of the corresponding ASes. Also, there are two parameters for choosing if the returning ASes are IPv4 or IPv6, or both. (for a detailed list, refer to the [doc](./Bias_in_IMPs.md) descrining our dataset)

Example request:
[https://ai4netmon.csd.auth.gr/api/rrc/rrc00](https://ai4netmon.csd.auth.gr/api/rrc/rrc00)

Example output:
```yaml
{
  "items": [
    {
      "ASN": "5392",
      "AS_rank_rank": "15184.0",
      "AS_rank_source": "RIPE",
      "AS_rank_longitude": "8.5215799323317",
      "AS_rank_latitude": "45.3317862341544",
      "AS_rank_numberAsns": "1.0",
      "AS_rank_numberPrefixes": "6.0",
      "AS_rank_numberAddresses": "16384.0",
      "AS_rank_total": "264.0",
      "AS_rank_customer": "0.0",
      "AS_rank_peer": "261.0",
      "AS_rank_provider": "3.0",
      "AS_rank_iso": "IT",
      "AS_rank_continent": "Europe",
      "is_personal_AS": "",
      "peeringDB_info_ratio": "Balanced",
      "peeringDB_info_traffic": "100.0",
      "peeringDB_info_scope": "Europe",
      "peeringDB_info_type": "NSP",
      "peeringDB_info_prefixes4": "100.0",
      "peeringDB_info_prefixes6": "50.0",
      "peeringDB_policy_general": "Open",
      "peeringDB_ix_count": "2.0",
      "peeringDB_fac_count": "1.0",
      "peeringDB_created": "2005-04-01T08:04:44Z",
      "is_in_peeringDB": "1.0",
      "AS_hegemony": "5.34168340933849e-06",
      "nb_atlas_probes_v4": "1.0",
      "nb_atlas_probes_v6": "0.0",
      "nb_atlas_anchors": "0.0",
      "is_ris_peer_v4": "1",
      "is_ris_peer_v6": "1",
      "is_routeviews_peer": "",
      "AS_rel_degree": "262.0",
      "cti_top": "",
      "cti_origin": "0.027920165776",
      "ASDB_C1L1": "Computer and Information Technology",
      "ASDB_C1L2": "Computer and Information Technology_Internet Service Provider (ISP)"
    },
    {
      "ASN": "5602",
      "AS_rank_rank": "2358.0",
      "AS_rank_source": "RIPE",
      "AS_rank_longitude": "10.5443750761765",
      "AS_rank_latitude": "45.0048952229435",
      "AS_rank_numberAsns": "10.0",
      "AS_rank_numberPrefixes": "143.0",
      "AS_rank_numberAddresses": "147712.0",
      "AS_rank_total": "330.0",
      "AS_rank_customer": "9.0",
      "AS_rank_peer": "318.0",
      "AS_rank_provider": "3.0",
      "AS_rank_iso": "IT",
      "AS_rank_continent": "Europe",
      "is_personal_AS": "",
      "peeringDB_info_ratio": "Mostly Outbound",
      "peeringDB_info_traffic": "10000.0",
      "peeringDB_info_scope": "Regional",
      "peeringDB_info_type": "Cable/DSL/ISP",
      "peeringDB_info_prefixes4": "100.0",
      "peeringDB_info_prefixes6": "5.0",
      "peeringDB_policy_general": "Open",
      "peeringDB_ix_count": "2.0",
      "peeringDB_fac_count": "3.0",
      "peeringDB_created": "2013-05-02T09:32:20Z",
      "is_in_peeringDB": "1.0",
      "AS_hegemony": "4.34473837590364e-05",
      "nb_atlas_probes_v4": "2.0",
      "nb_atlas_probes_v6": "0.0",
      "nb_atlas_anchors": "1.0",
      "is_ris_peer_v4": "1",
      "is_ris_peer_v6": "1",
      "is_routeviews_peer": "",
      "AS_rel_degree": "331.0",
      "cti_top": "",
      "cti_origin": "0.211582506271",
      "ASDB_C1L1": "Computer and Information Technology",
      "ASDB_C1L2": "Computer and Information Technology_Internet Service Provider (ISP)"
    }, ...
    }
```

### Distributions of IMPs `/distributions/{IMP}` 
It takes as input either (i) the name of an IMP or (ii) a list of ASNs and returns the distribution of All ASes and the distribution per feature, for the given IMP or ASN set.

Example request:
[https://ai4netmon.csd.auth.gr/api/distributions/Atlas](https://ai4netmon.csd.auth.gr/api/distributions/Atlas)

Options for `IMP` parameter are:
- `Atlas`
- `RIS`
- `RouteViews`
- `RIS&RouteViews` (combined peers of both IMPs)
- `rrc00` (the peers of the RIPE RIS route collector with id 00)
- ...
- `rrc26`
- `/custom/asn/?asn={ASN1}&asn={ASN2}&...&asn={ASNn}` (for a custom list of ASNs)
- `/custom/probe/?probe={probe1}&probe={probe2}&...&probe={proben}` (for a custom list of probes)

Example output:
```yaml

{"feature": "AS_rank_source",
 "xlabel":"RIR region",
 "ylabel":"Fraction",
 "bars":
     {"Atlas":
        {
         "RIPE":0.7034957897422812,
         "ARIN":0.11890788466445522,
         "APNIC":0.10334268946159735,
         "LACNIC":0.045930084205154376,
         "AFRINIC":0.028323551926511866
        }...
      }
}
```
  
### Bias cause of IMPs `/bias/cause/{IMP}` 

Example request:
[https://ai4netmon.csd.auth.gr/api/bias/cause/Atlas](https://ai4netmon.csd.auth.gr/api/bias/cause/Atlas)

Options for `IMP` parameter are:
- `Atlas`
- `RIS`
- `RouteViews`
- `RIS&RouteViews` (combined peers of both IMPs)
- `rrc/rrc00` (the peers of the RIPE RIS route collector with id 00)
- ...
- `rrc/rrc26`
- `asn/?asn={ASN1}&asn={ASN2}&...&asn={ASNn}` (for a custom list of ASNs)
- `probe/?probe={probe1}&probe={probe2}&...&probe={proben}` (for a custom list of probes)

Example output:
```yaml
{"Atlas":
    {"AS_rank_numberAsns":
        {
         "1.0-3.0":"-34.4331%",
         "3.0-9.0":"13.7526%",
         "9.0-26.0":"9.4607%",
         "26.0-76.0":"5.6389%",
         "76.0-222.0":"2.9668%",
         "222.0-654.0":"1.7342%",
         "654.0-1926.0":"0.4012%",
         "1926.0-5671.0":"0.3124%",
         "5671.0-16706.0":"0.0466%",
         "16706.0-49213.0":"0.1197%"
        }
    }
,...}
```
