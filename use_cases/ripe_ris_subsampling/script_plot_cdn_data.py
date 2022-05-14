#!/usr/bin/env python3
import json
from matplotlib import pyplot as plt
import numpy as np


FILE_CDN_DATA_RND300 = './cdn_data/rnd_atlas_300.json'
FILE_CDN_DATA_RND1000 = './cdn_data/rnd_atlas_1000.json'
FILE_CDN_DATA_SETS = './cdn_data/sub_select_atlas.json'
FILE_CDN_DATA_RND300_SIM = './cdn_data/rnd_atlas_sim_300.json'
FILE_CDN_DATA_RND1000_SIM = './cdn_data/rnd_atlas_sim_1000.json'
FILE_CDN_DATA_SETS_SIM = './cdn_data/simulated_atlas.json'

with open(FILE_CDN_DATA_RND300,'r') as f:
    d300 = json.load(f)
with open(FILE_CDN_DATA_RND1000,'r') as f:
    d1000 = json.load(f)
with open(FILE_CDN_DATA_SETS,'r') as f:
    d = json.load(f)
with open(FILE_CDN_DATA_RND300_SIM,'r') as f:
    d300_sim = json.load(f)
with open(FILE_CDN_DATA_RND1000_SIM,'r') as f:
    d1000_sim = json.load(f)
with open(FILE_CDN_DATA_SETS_SIM,'r') as f:
    d_sim = json.load(f)


def plot_rnd_sets(dd, set_size, color_line, color_area=None):
    '''
    plot random subsets
    '''
    err = np.empty((10,400))
    for i, k in enumerate(list(dd.keys())):
        pct = [w[0] for w in dd[k]]
        for j, v in enumerate(dd[k]):
            err[i,j] = v[1]
    err_mean = np.mean(err,axis=0)
    err_median = np.median(err,axis=0)
    err_std = np.std(err,axis=0)
    err_min = np.min(err,axis=0)
    err_max = np.max(err,axis=0)

    print('Mean error - Random (k={}): \t{}'.format(set_size, np.mean(err_mean[1:-2])))

    plt.plot(pct[1:-2],err_mean[1:-2], color=color_line, label='Random (k={}) - avg.'.format(set_size))
    step = 20
    plt.errorbar(pct[1::step],err_mean[1::step], 1.96/np.sqrt(10)*err_std[1::step],color=color_line,linestyle=' ')
    # plt.plot(pct[1:-2],err_median[1:-2],'--k')
    if color_area is not None:
        plt.fill_between(pct[1:-2], err_min[1:-2], err_max[1:-2], color=color_area, label='Random (k={}) - min/max'.format(set_size))


def plot_selected_sets(dd):
    '''
    plot selected subsets
    '''
    label_dict = {#'Atlas greedy 1000': 'Greedy (k=1000)',
                'Atlas greedy 300': 'Greedy (k=300)',
                # 'Atlas rnd 300',
                'Atlas All': 'Atlas All (k=3391)'}

    for k,v in dd.items():
        if k not in label_dict.keys():
            continue
        pct = [i[0] for i in v]
        err = [i[1] for i in v]
        print('Mean error - {}: \t{}'.format(label_dict[k],np.mean(err[1:-2])))
        plt.plot(pct[1:-2],err[1:-2],label=label_dict[k])


def apply_plotting_format(filename):
    '''
    common formatting
    '''
    FONTSIZE = 15
    # FONTSIZE_SMALL = 10
    plt.axis([-1,101,0,0.7])
    plt.xlabel('Percentile', fontsize=FONTSIZE)
    plt.ylabel('Relative error', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE, ncol=1, loc='upper right')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()



## generate plot actual atlas
print('Actual measurements')
plot_selected_sets(d)
plot_rnd_sets(d300, set_size=300, color_line='k', color_area=[0.9,0.9,0.9])
# plot_rnd_sets(d1000, set_size=1000, color_line='r', color_area=[0.9,0.8,0.8])
# plot_rnd_sets(d300, set_size=300, color_line='k')
# plot_rnd_sets(d1000, set_size=1000, color_line='r')
apply_plotting_format('./figures/fig_cdn_errors_atlas.png')

## generate plot simulated
print('Simulated measurements')
plot_selected_sets(d_sim)
plot_rnd_sets(d300_sim, set_size=300, color_line='k', color_area=[0.9,0.9,0.9])
# plot_rnd_sets(d1000_sim, set_size=1000, color_line='r', color_area=[0.9,0.8,0.8])
# plot_rnd_sets(d300_sim, set_size=300, color_line='k')
# plot_rnd_sets(d1000_sim, set_size=1000, color_line='r')
apply_plotting_format('./figures/fig_cdn_errors_simulated.png')



