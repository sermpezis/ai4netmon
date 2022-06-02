import pandas as pd
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib import pyplot as plt
import json
from Analysis.bias import bias_utils as bu

## data parameters
CDF_features = ['AS_rank_numberAsns', 'AS_rank_numberPrefixes', 'AS_rank_numberAddresses','AS_rank_total','AS_rank_customer', 
        'AS_rank_peer', 'AS_rank_provider', 'peeringDB_ix_count', 'peeringDB_fac_count', 'AS_hegemony']

Histogram_features = ['AS_rank_source', 'AS_rank_continent', 'peeringDB_info_ratio','peeringDB_info_traffic', 
        'peeringDB_info_scope', 'peeringDB_info_type','peeringDB_policy_general', 'is_personal_AS']

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
    FEATURE_NAMES_DICT = bu.get_features_dict_for_visualizations() 
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


def plot_all(dict_network_sets, filename_format, save_json=False, show_plot=False):
    '''
    Plots the distibutions (CDF or Histogram) for all "CDF_features" and "Histogram_features" for the given data. 
    The set of features to be plotted are defined in this file (it's not from the given data)
    It saves the figures, and depending on the arguments it can show the figures and/or save the plot data in json.

    :param  dict_network_sets:  (dict {str:pandas.Series}) keys are the name of the network set (e.g., 'All ASes' or 'RIPE Atlas') 
                                and values the corresponding dataframe (i.e., with only the specific rows)
    :param  filename_format:    (str) the filename template to be used for saving the figures (and json files)
    :param  save_json:          (boolean) if true it saves the plot data in json
    :param  show_plot:          (boolean) if true it also shows the plots when generating them
    '''
    for feature in CDF_features:
        data = generate_plot_json(feature, 'CDF', dict_network_sets)
        filename_no_ext = filename_format.format('CDF',feature)
        if save_json:
            with open(filename_no_ext+'.json', 'w') as f:
                json.dump(data,f)
        plot_cdf(data, dict_network_sets, filename_no_ext, show_plot=show_plot)

    for feature in Histogram_features:
        data = generate_plot_json(feature, 'histogram', dict_network_sets)
        filename_no_ext = filename_format.format('Histogram',feature)
        if save_json:
            with open(filename_no_ext+'.json', 'w') as f:
                json.dump(data,f)
        plot_histogram(data, dict_network_sets, filename_no_ext, show_plot=show_plot)