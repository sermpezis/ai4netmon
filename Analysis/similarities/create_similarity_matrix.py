import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from dotenv import load_dotenv

load_dotenv()
key = 'ASN2ASN_FILENAME'
ASN2ASN_FILENAME = os.getenv(key)


def load_dict_from(filename):
    """
    :param filename: The local path where the asn2asn.json file is stored
    :return: a dictionary of the given .json file
    """
    with open(filename) as json_file:
        data = json.load(json_file)
    return data

def creating_list_containing_all_pairs(data_dict):
    """
    :param data_dict: The json file containing as key an ASn and as values the ASns that the key ASn exchange information
    :return: A list of lists, where each list contains all possible pairs for each specific ASn(key)
    """
    for asn in data_dict:
        new_pairs = []
        flag = True
        for asn2 in data_dict[asn]:
            if flag:
                new_pairs.append(asn)
                new_pairs.append(asn2)
                flag = False
            else:
                new_pairs.append(asn2)
        list_containing_all_possible_pairs.append(new_pairs)
    return list_containing_all_possible_pairs

def similarity_function(data, data2):
    """
    :param data: The ASn1
    :param data2: The ASn2
    :return: The user's chosen similarity metric (Gini, Gini_strict, Cosine_similarity) for a pair of ASNs (keys)
    """
    metric = 'Gini'
    inter = np.intersect1d(data, data2)
    union = np.union1d(data, data2)
    if metric == 'Gini':
        similarity = len(inter)/len(union)
    elif metric == 'Gini_strict':
        inter_common_values = [x for x in inter if (data.index(x) & data2.index(x)) > 0]
        similarity = len(inter_common_values)/len(union)
    elif metric == 'Cosine_similarity':
        values1 = [x if x in union else 0 for x in data]
        values2 = [x if x in union else 0 for x in data2]
        if len(values1) > len(values2):
            new_values2 = np.array([values2[i] if i < len(values2) else 0 for i in range(0, len(values1))])
            new_similarity = cosine_similarity(np.array(values1).reshape(1, -1), new_values2.reshape(1, -1))
            similarity = new_similarity[0][0]
        else:
            new_values1 = np.array([values1[i] if i < len(values1) else 0 for i in range(0, len(values2))])
            new_similarity = cosine_similarity(new_values1.reshape(1, -1), np.array(values2).reshape(1, -1))
            similarity = new_similarity[0][0]
    else:
        raise Exception('Not defined similarity method')
    return similarity

# calculation of the similarity matrix
data_dict = load_dict_from(ASN2ASN_FILENAME)
list_containing_all_possible_pairs = []
list_containing_all_possible_pairs = creating_list_containing_all_pairs(data_dict)

similarity_dict = {}
for i in range(0, len(list_containing_all_possible_pairs)-1):
    asn1 = list_containing_all_possible_pairs[i]
    for j in range(i+1, len(list_containing_all_possible_pairs)):
        asn2 = list_containing_all_possible_pairs[j]
        similarity_dict[asn1[0], asn2[0]] = similarity_function(asn1, asn2)
    print(similarity_dict)