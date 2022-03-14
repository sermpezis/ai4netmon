import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.ticker import MaxNLocator

GRID = True
LOCATOR = 4
FONTSIZE = 15
FIGSIZE = (15, 15)
COLOR = "k"
plot_kwargs = {'dpi': 300, 'bbox_inches': 'tight'}

PLOT_KIND = [None, "operators", "researchers", "CP", "DP"]
# choose kind to get plots. None is for regular plots, and the other strings stand
# for the filtering is going to be made on plots
kind = PLOT_KIND[0]

if kind is None:
    PLOT_FNAME = "fig_survey_fig{}"
else:
    PLOT_FNAME = "fig_survey_only_{}_fig{}"


def add_percentages_to_plot(ax, total):
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width() / total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height() / 2
        ax.annotate(percentage, (x, y))


def barh_plots(df, basic_cols, count_for_png, save_filename, kind):
    # plot basic columns
    for i in range(len(basic_cols)):
        total = df[basic_cols[i]].count().sum()
        plt.figure()
        plt.grid(GRID)
        plt.rcParams.update({'font.size': FONTSIZE})

        plt.title(basic_cols[i])
        ax = df[basic_cols[i]].str.split(r',\s*(?=[^)]*(?:\(|$))', expand=True).stack().value_counts().plot(kind='barh',
                                                                                                            figsize=FIGSIZE,
                                                                                                            color='k')

        add_percentages_to_plot(ax, total)

        locator = matplotlib.ticker.MultipleLocator(LOCATOR)
        ax.xaxis.set_major_locator(locator)
        if kind is None:
            plt.savefig(save_filename.format(count_for_png), **plot_kwargs)
        else:
            plt.savefig(save_filename.format(kind, count_for_png), **plot_kwargs)
        plt.close()
        count_for_png += 1
    return count_for_png


def barh_plots_simple_answers_use_cases(df, cols, count_for_png, save_filename, default_answers, other_answers,
                                        true_answers, title, kind):
    total = 0
    for i in range(len(cols)):
        total += df[cols[i]].count().sum()

    counts = []

    i = 0
    while i < len(default_answers):
        sum = 0
        for col in cols:
            df[col] = df[col].astype(str)
            sum += df[col].str.count(default_answers[i]).sum()

        counts.append(sum)
        i += 1
    data = []

    for i in range(len(counts)):
        data.append((default_answers[i], counts[i]))

    if other_answers != None:
        j = 0
        other_sum = 0
        while j < len(other_answers):

            for col in cols:
                other_sum += df[col].str.count(other_answers[j]).sum()

            j += 1

        data.append(("Other", other_sum))

    df_loc = pd.DataFrame(data, columns=["Answers", "Question"])
    df_loc.index = true_answers

    plt.figure()
    plt.rcParams.update({'font.size': FONTSIZE})

    ax = df_loc.plot(kind='barh', title=title, figsize=FIGSIZE, color='k')

    locator = matplotlib.ticker.MultipleLocator(LOCATOR)
    ax.xaxis.set_major_locator(locator)

    add_percentages_to_plot(ax, total)

    if kind is None:
        plt.savefig(save_filename.format(count_for_png), **plot_kwargs)
    else:
        plt.savefig(save_filename.format(kind, count_for_png), **plot_kwargs)
    plt.close()
    count_for_png += 1

    return count_for_png


def barh_plots_multiple_answers_use_cases(df, cols, count_for_png, save_filename, default_answers, other_answers,
                                          true_answers, title, kind):
    total = 0
    for i in range(len(cols)):
        total += df[cols[i]].count().sum()

    counts_d = []

    i = 0
    while i < len(default_answers):
        sum = 0
        for col in cols:
            sum += df[col].str.count(default_answers[i]).sum()

        counts_d.append(sum)
        i += 1

    j = 0

    other_sum_idxs = []
    while j < len(other_answers):
        idx = []
        for col in cols:
            idx.append(df.index[df[col].str.contains(other_answers[j], na=False)].tolist())
        other_sum_idxs.append(idx)
        j += 1

    # print(other_sum_idxs)
    flat_other_sum = [item for sublist in other_sum_idxs for item in sublist]
    flat_other_sum_idxs = [item for sublist in flat_other_sum for item in sublist]

    flat_other_sum_idxs = list(set(flat_other_sum_idxs))
    other_sum = len(flat_other_sum_idxs)
    data = []

    for i in range(len(counts_d)):
        data.append((default_answers[i], counts_d[i]))

    data.append(("Other", other_sum))

    df_bias = pd.DataFrame(data, columns=["Answers", "Question"])

    df_bias.index = true_answers

    plt.figure()
    plt.rcParams.update({'font.size': FONTSIZE})

    ax = df_bias.plot(kind='barh', title=title, figsize=FIGSIZE, color='k')

    locator = matplotlib.ticker.MultipleLocator(LOCATOR)
    ax.xaxis.set_major_locator(locator)

    add_percentages_to_plot(ax, total)
    if kind is None:
        plt.savefig(save_filename.format(count_for_png), **plot_kwargs)
    else:
        plt.savefig(save_filename.format(kind, count_for_png), **plot_kwargs)
    plt.close()
    count_for_png += 1

    return count_for_png


def hbar_networks(df, columns_of_networks, count_for_png, save_filename, kind):
    for network in columns_of_networks:
        df[network] = df[network].astype(str)

    networks = []
    tier = [col for col in columns_of_networks if 'Tier' in col]
    multin = [col for col in columns_of_networks if 'Multi-national' in col]
    reg = [col for col in columns_of_networks if 'Regional' in col]
    cont = [col for col in columns_of_networks if 'Content Distribution' in col]
    cloud = [col for col in columns_of_networks if 'Cloud provider' in col]
    mob = [col for col in columns_of_networks if 'Mobile Access' in col]
    ixp = [col for col in columns_of_networks if 'IXP' in col]
    enter = [col for col in columns_of_networks if 'Enterprise' in col]
    oth = [col for col in columns_of_networks if 'Other' in col]

    networks.append(tier)
    networks.append(multin)
    networks.append(reg)
    networks.append(cont)
    networks.append(cloud)
    networks.append(mob)
    networks.append(ixp)
    networks.append(enter)
    networks.append(oth)

    j = 0
    all_networks = []
    while j < len(networks):
        i = 0

        while i < len(networks[j]):
            sum1 = 0
            sum2 = 0
            for col in networks[j]:
                sum1 += df[col].str.count("From").sum()
                sum2 += df[col].str.count("To").sum()
            i += 1
        all_networks.append((networks[j][0], sum1, sum2))
        j += 1

    df_networks = pd.DataFrame(all_networks, columns=['Network', 'From', 'To'])
    df_networks = df_networks.set_index('Network')

    df_networks.index = df_networks.index.str.split('\[').str[-1].str.strip()
    df_networks.index = df_networks.index.str[:-1]

    plt.figure()
    plt.rcParams.update({'font.size': FONTSIZE})

    ax = df_networks.plot(kind='barh', figsize=FIGSIZE, title='Network Type')

    locator = matplotlib.ticker.MultipleLocator(10)
    ax.xaxis.set_major_locator(locator)
    ax.set(ylabel=None)
    if kind is None:
        plt.savefig(save_filename.format(count_for_png), **plot_kwargs)
    else:
        plt.savefig(save_filename.format(kind, count_for_png), **plot_kwargs)
    plt.close()
    count_for_png += 1
    return count_for_png


def hbar_multibar(df, cols, count_for_png, save_filename, responses, title, kind):
    values = []
    for c in cols:
        values.append([df[c].astype(str).str.count(r).sum() for r in responses])

    data = []
    for i in range(len(values)):
        data.append([cols[i]] + [values[i][j] for j in range(len(responses))])

    df_plot = pd.DataFrame(data, columns=['Question'] + responses)
    df_plot = df_plot.set_index('Question')

    df_plot.index = df_plot.index.str.split('\[').str[-1].str.strip()
    df_plot.index = df_plot.index.str[:-1]

    plt.figure()
    plt.rcParams.update({'font.size': FONTSIZE})

    ax = df_plot.plot(kind='barh', figsize=FIGSIZE, title=title)
    ax.set(ylabel=None)

    locator = matplotlib.ticker.MultipleLocator(LOCATOR)
    ax.xaxis.set_major_locator(locator)
    if kind is None:
        plt.savefig(save_filename.format(count_for_png), **plot_kwargs)
    else:
        plt.savefig(save_filename.format(kind, count_for_png), **plot_kwargs)
    plt.close()
    count_for_png += 1
    return count_for_png
