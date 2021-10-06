'''
TODO: if you don't know about environment variables, check e.g. https://www.twilio.com/blog/environment-variables-python

I have added an example .env file in the parent folder
You can edit it locally (i.e., put the correct path to file in your local pc) but do not commit/push it
'''
import os 
# requires installing the following: pip3 install python-dotenv
from dotenv import load_dotenv


load_dotenv()
ASN2ASN_FILENAME = os.getenv('ASN2ASN_FILENAME')


### pseudocode for the calculation of the similariity matrix
data_dict = load_dict_from(ASN2ASN_FILENAME)
list_of_asns = list(data_dict.keys())
for asn1, asn2 in all_pairs_of_asns_in_list_of_asns:
	similarity(asn1, asn2) = similarity_function(data_dict[asn1],data_dict[asn2])



### pseudocode for the different similariity functions to be implemented
#1 - metric: gini
inter = intersection(data_dict[asn1].keys(), data_dict[asn2].keys())
union = union(data_dict[asn1].keys(), data_dict[asn2].keys())
similarity_function1 = len(inter)/len(union)

#2 - metric: "gini-strict"
inter_common_values = [x for x in inter if data_dict[asn1][x]==data_dict[asn1][x]]
similarity_function2 = len(inter_common_values)/len(union)

#3 - metric: cosine similarity
values1 = [data_dict[asn1].get(x,0) for x in union]
values2 = [data_dict[asn2].get(x,0) for x in union]
similarity_function3 = cosine_similarity(values1, values2)