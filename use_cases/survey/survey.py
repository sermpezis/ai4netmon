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

# take columns for each question
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

# basic cols
count_for_png = spm.barh_plots(df, basic_cols, count_for_png, PLOT_FNAME)

# Measurment
title = 'Measurement types'
other_answers = None
default_answers = ["Control", "Data", "know"]
true_answers = ["Control-plane (BGP tables and messages/updates )", "Data-plane (ping, traceroute)", "I don't know"]
cols = measurement_cols
count_for_png = barh_plots_simple_answers_use_cases(df, cols, count_for_png, PLOT_FNAME, default_answers, other_answers, true_answers, title)

# Scope
title = 'Scope'
other_answers = ["update", "bgp", "AS", "PTR", "Cybersecurity", "Uptime", "Rpki", "User's"]
default_answers = ['Paths', 'Reachability', 'Latency', 'Throughput']
true_answers = ['Paths, routing policies, topology (e.g., BGP messages, traceroutes)', 'Reachability', 'Latency',
                      'Throughput', 'Other']
cols = scope_cols
count_for_png = barh_plots_multiple_answers_use_cases(df, cols, count_for_png, PLOT_FNAME, default_answers, other_answers, true_answers, title)

# Infrastructure
title = 'inf'
other_answers = ["rpki", "NLRING", "dig", "MTR.sh", "bgp.tools", "whois", "ITDK", "Speedtest.net", "isolario",
                     "PTR", "Eyes", "looking-glass", "Radar", "PeeringDB", "Euro-IX", "PHC", "scans.io"]
default_answers = ["RIS", "Atlas", "RIPEstat", "bgp.he.net", "RouteViews", "CAIDA’s", "Ark", "M-lab", "Custom"]
true_answers = ["RIPE RIS", "RIPE Atlas", "RIPEstat", "bgp.he.net", "RouteViews", "CAIDA’s BGPStream", "Ark",
                    "M-lab", "Custom of proprietary measurement platform/service", "Other"]
cols = inf_cols
count_for_png = barh_plots_multiple_answers_use_cases(df, cols, count_for_png, PLOT_FNAME, default_answers, other_answers, true_answers, title)

# Location
title = 'loc'
other_answers = ["Depend", "between", "Hyper-local", "Company", "physical"]
default_answers = ["City-level", "Country-level", "Continent-level", "Global-level", "know"]
true_answers = ["City-level", "Country-level", "Continent-level", "Global-level",
                                              "I don't know", "Other"]
cols = loc_cols
count_for_png = barh_plots_simple_answers_use_cases(df, cols, count_for_png, PLOT_FNAME, default_answers, other_answers, true_answers, title)

# Networks
count_for_png = spm.hbar_networks(df, columns_of_networks, count_for_png, PLOT_FNAME)

# bias
title = 'Is there any kind of bias in the measurement data collected for this use case?'
other_answers = None
default_answers = ["know", "No", "Yes"]
true_answers = ["I don't know", "No/Probably no", "Yes/Probably yes"]
cols = bias_cols
count_for_png = barh_plots_simple_answers_use_cases(df, cols, count_for_png, PLOT_FNAME, default_answers, other_answers, true_answers, title)

# if yes for bias
title = '[Optional] If yes, the bias is with respect to...?'
other_answers = ["partial", "imho", "nobody", "representative", "ourselves", "stub.", "Scarcity", "IPv6",
                     "filtering", "Access", "ie", "Atlas", "view"]
default_answers = ["Geography", "types"]
true_answers = ["Geography / location (e.g., geographic locations)",
                     "Network types (e.g., eyeball, transit, CDNs)", "Other"]
cols = bias2_cols
count_for_png = barh_plots_multiple_answers_use_cases(df, cols, count_for_png, PLOT_FNAME, default_answers, other_answers, true_answers, title)

# indication of bias
title = 'Is a good indication of bias helpful?'
other_answers = None
default_answers = ["1.0", "2.0", "3.0", "4.0", "5.0"]
true_answers = default_answers
cols = numerical_cols
count_for_png = barh_plots_simple_answers_use_cases(df, cols, count_for_png, PLOT_FNAME, default_answers, other_answers, true_answers, title)

# trigger
responses = ["Would it trigger use of measurement data or infrastructure?", "Are you concerned about bias in the data?"]
title = 'Which of the following types of events would trigger you to use measurement/monitoring data '
cols = columns_of_trigger
count_for_png = spm.hbar_multibar(df, cols, count_for_png, PLOT_FNAME, responses, title)

# rest basic cols
count_for_png = spm.barh_plots(df, basic_cols2, count_for_png, PLOT_FNAME)

# importance
responses = ["1.0", "2.0", "3.0", "4.0", "5.0"]
title = 'How important would be the following advances in Internet monitoring/measurements for you?'
cols = columns_of_importance
count_for_png = spm.hbar_multibar(df, cols, count_for_png, PLOT_FNAME, responses, title)



