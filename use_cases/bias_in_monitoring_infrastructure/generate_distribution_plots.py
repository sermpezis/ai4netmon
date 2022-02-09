#!/usr/bin/env python3
#
# Author: Pavlos Sermpezis (https://sites.google.com/site/pavlossermpezis/)
#
import pandas as pd
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib import pyplot as plt
import json

SHOW_PLOTS = False
SAVE_TO_JSON = False



## data parameters
DATA_FILE = '../../data/aggregate_data/asn_aggregate_data_20211201.csv'
RIPE_RIS_DATA_FILE = '../../data/misc/RIPE_RIS_peers_ip2asn.json'
ROUTEVIEWS_FNAME = '../../data/misc/RouteViews_peers.json'
SAVE_PLOTS_FNAME_FORMAT = './figures/Fig_{}_{}' # the first brackets to be filled with 'CDF' or 'Histogram', and the second with the feature name

CDF_features = ['AS_rank_numberAsns', 'AS_rank_numberPrefixes', 'AS_rank_numberAddresses','AS_rank_total','AS_rank_customer', 
        'AS_rank_peer', 'AS_rank_provider', 'peeringDB_ix_count', 'peeringDB_fac_count', 'AS_hegemony']

Histogram_features = ['AS_rank_source', 'AS_rank_continent', 'peeringDB_info_ratio','peeringDB_info_traffic', 
        'peeringDB_info_scope', 'peeringDB_info_type','peeringDB_policy_general', 'is_personal_AS']


FEATURE_NAMES_DICT = {
    # Location-related info
    'AS_rank_source': 'RIR region',
    'AS_rank_iso': 'Location\n (country)',
    'AS_rank_continent': 'Location\n (continent)',
    # network size info
    'AS_rank_numberAsns': 'Customer cone\n (#ASNs)', 
    'AS_rank_numberPrefixes': 'Customer cone\n (#prefixes)',
    'AS_rank_numberAddresses': 'Customer cone\n (#addresses)',
    'AS_hegemony': 'AS hegemony',
    # Topology info
    'AS_rank_total': '#neighbors\n (total)',
    'AS_rank_peer': '#neighbors\n (peers)', 
    'AS_rank_customer': '#neighbors\n (customers)', 
    'AS_rank_provider': '#neighbors\n (providers)',
    # IXP related
    'peeringDB_ix_count': '#IXPs\n (PeeringDB)', 
    'peeringDB_fac_count': '#facilities\n (PeeringDB)', 
    'peeringDB_policy_general': 'Peering policy\n (PeeringDB)',
    # Network type
    'peeringDB_info_type': 'Network type\n (PeeringDB)',
    'peeringDB_info_ratio': 'Traffic ratio\n (PeeringDB)',
    'peeringDB_info_traffic': 'Traffic volume\n (PeeringDB)', 
    'peeringDB_info_scope': 'Scope\n (PeeringDB)',
    'is_personal_AS': 'Personal ASN', 
}


## plotting parameters
FONTSIZE = 15
LINEWIDTH = 2
MARKERSIZE = 10



def generate_plot_json(feature, plot_type, dict_network_sets):
    '''
    Generates a dict with all the needed information for the plots. The dict can be written to a json.

    :param  feature:    (str) the name of the feature, i.e., column of the datafame, to be plotted
    :param  plot_type:  (str) 'CDF' or 'histogram'
    :param  dict_network_sets:  (dict {str:pandas.Series}) keys are the name of the network set (e.g., 'All ASes' or 'RIPE Atlas') 
                                and values the corresponding dataframe (i.e., with only the specific rows)
    :return:    (dict) with all the needed information for plotting the given feature
    '''
    data = dict()
    data['feature'] = feature
    data['xlabel'] = FEATURE_NAMES_DICT[feature]
    if plot_type == 'CDF':
        data['ylabel'] = 'CDF'
        data['xscale'] = 'log'
        data['yscale'] = 'linear'
        data['curves'] = dict()
        for set_name, df in dict_network_sets.items():
            cdf = ECDF(df[feature].dropna())
            data['curves'][set_name] = dict()
            data['curves'][set_name]['x'] = cdf.x.tolist()
            data['curves'][set_name]['y'] = cdf.y.tolist()
    elif plot_type == 'histogram':
        data['ylabel'] = 'Fraction'
        data['bars'] = dict()
        for set_name, df in dict_network_sets.items():
            x = df[feature].replace('',np.nan).value_counts()
            x = x/sum(x)
            data['bars'][set_name] = dict(x)
    else:
        raise ValueError

    return data



def plot_cdf(data, dict_network_sets, filename, show_plot=False):
    '''
    Plots a CDF for the given data, under the defined plotting parameters. 
    It shows and saves the figure.
    '''
    sets_str = list(dict_network_sets.keys())
    for set_name in sets_str:
        plt.plot(data['curves'][set_name]['x'], data['curves'][set_name]['y'], linewidth=LINEWIDTH)
    plt.xlabel(data['xlabel'], fontsize=FONTSIZE)
    plt.ylabel(data['ylabel'], fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.xscale(data['xscale'])
    plt.yscale(data['yscale'])
    plt.legend(sets_str, fontsize=FONTSIZE)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename+'.png')
    if show_plot:
        plt.show()
    plt.close()




def plot_histogram(data, dict_network_sets, filename, show_plot=False):
    '''
    Plots a histogram for the given data, under the defined plotting parameters. 
    It shows and saves the figure.
    '''
    sets_str = list(dict_network_sets.keys())
    X = pd.concat([pd.Series(data['bars'][set_name]) for set_name in sets_str], axis=1)
    X.plot.bar()
    plt.gca().set_ylim([0, 1])
    plt.xlabel(data['xlabel'], fontsize=FONTSIZE)
    plt.ylabel(data['ylabel'], fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(sets_str, fontsize=FONTSIZE)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename+'.png')
    if show_plot:
        plt.show()
    plt.close()


if __name__ == '__main__':
    ## load data, create dataframes
    df = pd.read_csv(DATA_FILE, header=0, index_col=0)
    df['is_personal_AS'].fillna(0, inplace=True)

    df_atlas = df.loc[(~df['nb_atlas_probes_v4'].isna())|(~df['nb_atlas_probes_v4'].isna()),:]

    ris_asns = pd.read_json(RIPE_RIS_DATA_FILE, orient='index')[0].unique().tolist()
    ris_asns = [i for i in ris_asns if i in df.index]
    df_ris = df.loc[ris_asns,:]

    routeviews_asns = pd.read_json(ROUTEVIEWS_FNAME)[0].unique().tolist()
    routeviews_asns = [i for i in routeviews_asns if i in df.index]
    df_rv = df.loc[routeviews_asns,:]


    ## generate json (and plots)
    dict_network_sets = {'All ASes': df, 'RIPE Atlas': df_atlas, 'RIPE RIS': df_ris, 'RouteViews': df_rv}
    for feature in CDF_features:
        data = generate_plot_json(feature, 'CDF', dict_network_sets)
        filename_no_ext = SAVE_PLOTS_FNAME_FORMAT.format('CDF',feature)
        if SAVE_TO_JSON:
            with open(filename_no_ext+'.json', 'w') as f:
                json.dump(data,f)
        plot_cdf(data, dict_network_sets, filename_no_ext, show_plot=SHOW_PLOTS)
    for feature in Histogram_features:
        data = generate_plot_json(feature, 'histogram', dict_network_sets)
        filename_no_ext = SAVE_PLOTS_FNAME_FORMAT.format('Histogram',feature)
        if SAVE_TO_JSON:
            with open(filename_no_ext+'.json', 'w') as f:
                json.dump(data,f)
        plot_histogram(data, dict_network_sets, filename_no_ext, show_plot=SHOW_PLOTS)