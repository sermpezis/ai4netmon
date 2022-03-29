import numpy as np
import pandas as pd
import json
import random
from matplotlib import pyplot as plt
from ai4netmon.Analysis.bias import bias_utils as bu
from ai4netmon.Analysis.bias import radar_chart
from ai4netmon.Analysis.bias import generate_distribution_plots as gdp
import os




BIAS_CSV_FNAME = './data/bias_values_random_samples.csv'
FIG_RADAR_FNAME_FORMAT = './figures/Fig_radar_RIPE_Atlas_sampling.png'


bias_df = pd.read_csv(BIAS_CSV_FNAME, header=0, index_col=0)


NB_SAMPLES_PLOT = [10, 20, 100]
bias_df_avg = bias_df[['RIPE Atlas (all)']]
for nb in NB_SAMPLES_PLOT:
    cols = [c for c in bias_df.columns if c.startswith('RIPE Atlas{}_'.format(nb))]
    bias_df_avg['RIPE Atlas ({} samples)'.format(nb)] = bias_df[cols].mean(axis=1)
print(bias_df_avg.round(2))
radar_chart.plot_radar_from_dataframe(bias_df_avg, colors=None, frame='polygon', legend_loc=(0.85,0.95), save_filename=FIG_RADAR_FNAME_FORMAT)
