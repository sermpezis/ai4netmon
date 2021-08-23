import json
from matplotlib import pyplot as plt

INPUT_FILENAME = 'select_monitors_10000.json'


'''
Load data
'''
with open(INPUT_FILENAME, 'r') as f:
    DATA = json.load(f)
    #  DATA.keys()  = 'asns naive', 'dist naive', 'asns greedy', 'dist greedy'



'''
Define ploting options
'''
linewidth = 3.0
markersize = 10.0
fontsize = 20
common_plot_kwargs = {'linewidth':linewidth, 'markersize':markersize, 'markeredgewidth':linewidth}#, 'markerfacecolor':'None'


'''
Plots
'''


def plot_nb_monitors_vs_dist(DATA, MAX_X):
    x_vector = list(range(0,MAX_X+1))
    y_vector_naive = [DATA['dist naive'][i] / DATA['dist naive'][0] for i in range(MAX_X+1)]
    y_vector_greedy = [DATA['dist greedy'][i] / DATA['dist greedy'][0] for i in range(MAX_X+1)]
    plt.plot(x_vector, y_vector_naive, '--k', x_vector, y_vector_greedy, '-k', **common_plot_kwargs)
    plt.legend(['Sorted', 'Greedy'], fontsize=fontsize)
    plt.xlabel('# new monitors',fontsize=fontsize)
    plt.ylabel('distance (normalized)',fontsize=fontsize)
    plt.title('total distance = {}'.format(DATA['dist naive'][0]))
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid(True)  
    plt.tight_layout()
    # plt.subplots_adjust(left=0.17, bottom=0.17)
    plt.savefig('fig_set_monitors_vs_distance_{}.png'.format(MAX_X))
    plt.close()

plot_nb_monitors_vs_dist(DATA, 10)
plot_nb_monitors_vs_dist(DATA, 100)
plot_nb_monitors_vs_dist(DATA, 1000)
plot_nb_monitors_vs_dist(DATA, 10000)



def plot_monitors_vs_dist(DATA, MAX_X):
    x_vector = list(range(1,MAX_X+1))
    y_vector_naive = [DATA['dist naive'][i-1]-DATA['dist naive'][i] for i in range(1,MAX_X+1)]
    y_vector_greedy = [DATA['dist greedy'][i-1]-DATA['dist greedy'][i] for i in range(1,MAX_X+1)]
    plt.semilogy(x_vector, y_vector_naive, '--k', x_vector, y_vector_greedy, '-k', **common_plot_kwargs)
    plt.legend(['Sorted', 'Greedy'], fontsize=fontsize)
    plt.xlabel('i^{th} monitor',fontsize=fontsize)
    plt.ylabel('improvement by i^{th} monitor',fontsize=fontsize)
    # plt.title('total distance = {}'.format(DATA['dist naive'][0]))
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid(True)  
    plt.tight_layout()
    # plt.subplots_adjust(left=0.17, bottom=0.17)
    plt.savefig('fig_monitors_vs_distance_decrease_{}.png'.format(MAX_X))
    plt.close()

plot_monitors_vs_dist(DATA, 10)
plot_monitors_vs_dist(DATA, 100)
plot_monitors_vs_dist(DATA, 1000)
plot_monitors_vs_dist(DATA, 10000)