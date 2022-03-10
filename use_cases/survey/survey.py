import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.ticker import MaxNLocator
import survey_plot_methods as spm

# # set parameters (global)
PLOT_FNAME = "fig_survey_fig{}"
count_for_png = 0  # return it every time we call a plot function

df = pd.read_csv('SurveyAnswers.csv')

# basic cols
basic_cols = ['What is your main professional role?', 'Which term(s) would best characterize your organization?',
              'In which continent(s) does your organization operate (or is located)?',
              'What is your knowledge or experience with Internet measurements/monitoring?']

basic_cols2 = ['How many platforms do you typically use when you do measurements?',
               'How many monitors / vantage points (VPs) do you typically use in your measurements?']

# take network columns
columns_of_networks = [col for col in df.columns if 'Network types' in col]
columns_of_importance = [col for col in df.columns if 'How important would be' in col]
columns_of_trigger = [col for col in df.columns if 'Which of the following types of events would trigger' in col]
numerical_cols = [col for col in df.columns if 'good indication of the bias?' in col]
measurement_cols = [col for col in df.columns if 'Measurement' in col]
scope_cols = [col for col in df.columns if 'Scope' in col]
inf_cols = [col for col in df.columns if 'Infrastructure / platforms' in col]
bias_cols = [col for col in df.columns if 'Is there any kind of bias in the measurement data collected for this use case?' in col]
bias2_cols = [col for col in df.columns if 'If yes, the bias is with respect to' in col]
loc_cols = [col for col in df.columns if 'Location' in col]


# count_for_png = spm.barh_plots(df, basic_cols, count_for_png, PLOT_FNAME)
# count_for_png = spm.barh_plots_measurement(df, measurement_cols, count_for_png, PLOT_FNAME)
# count_for_png = spm.barh_plots_scope(df, scope_cols, count_for_png, PLOT_FNAME)
# count_for_png = spm.barh_plots_inf(df, inf_cols, count_for_png, PLOT_FNAME)
# count_for_png = spm.barh_plots_loc(df, loc_cols, count_for_png, PLOT_FNAME)
# count_for_png = spm.hbar_networks(df, columns_of_networks, count_for_png, PLOT_FNAME)
# count_for_png = spm.barh_plots_bias(df, bias_cols, count_for_png, PLOT_FNAME)
# count_for_png = spm.barh_plots_ifyesbias(df, bias2_cols, count_for_png, PLOT_FNAME)
# count_for_png = spm.barh_plots_num(df, numerical_cols, count_for_png, PLOT_FNAME)
# count_for_png = spm.hbar_trigger(df, columns_of_trigger, count_for_png, PLOT_FNAME)
# count_for_png = spm.barh_plots(df, basic_cols2, count_for_png, PLOT_FNAME)
# count_for_png = spm.hbar_importance(df, columns_of_importance, count_for_png, PLOT_FNAME)

responses = ["1.0", "2.0", "3.0", "4.0", "5.0"]
title = 'How important would be the following advances in Internet monitoring/measurements for you?'
cols = columns_of_importance
count_for_png = spm.hbar_multibar(df, cols, count_for_png, PLOT_FNAME, responses, title)

responses = ["Would it trigger use of measurement data or infrastructure?", "Are you concerned about bias in the data?"]
title = 'Which of the following types of events would trigger you to use measurement/monitoring data '
cols = columns_of_trigger
count_for_png = spm.hbar_multibar(df, cols, count_for_png, PLOT_FNAME, responses, title)

