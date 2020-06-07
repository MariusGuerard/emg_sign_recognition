import numpy as np
import matplotlib.pyplot as plt

def bar_plot_scatter_comp(list_vec, x_labels, title, legend, save_name=None, stats_list=None, xlabel='Model', ylabel='Acc'):
    """Bar plot of each elements of list_vec. 
    (For example if list_vec = [vec_A, vec_B] it will plot 2 bars 
    for each time point along error bar and superimposed with scatterplot).
    Args:
        - list_vec ([np.array]): the observed values of dimension N_samples, N_time.
        - x_labels ([int]): list of integers specifying the x values.
        - title (str): Title of the plot.
        - legend ([str]): one string per vector of list_vec.
        - save_name (str): Save the figure if a path is provided.
        - stats_list ([float]): One p-value for each time point corresponding 
        to the prob. of H0: "<list_vec[0]> = <list_vec[1]>".
    """
    ### Function to return a vector (of length 'size') made of random number between -1 and 1.
    rand_m1 = lambda size: np.random.random(size) * 2 - 1

    ### Plots parameters.
    N_serie = len(list_vec)
    label_loc = np.arange(len(x_labels))  # the label locations
    w = 0.5  # the width of the bars
    colors = ['b', 'r', 'gray']
    center_bar = [w * (i - 0.5) for i in range(N_serie)]

    ### Plots values.
    avg_conc = [np.nanmean(conc, axis=0) for conc in list_vec]
    std_conc = [np.nanstd(conc, axis=0) for conc in list_vec]
        
    ### Plotting the data.
    fig, ax = plt.subplots(figsize=(20, 15))
    
    for i, conc in enumerate(list_vec):
        label_i = legend[i] + ' (n = {})'.format(len(conc))
        rects = ax.bar(label_loc + center_bar[i], avg_conc[i], w, label=label_i,
                       yerr=std_conc[i], color=colors[i], alpha=0.6)
        # Add the single point with a scatter distributed randomly across whole width of bar.
        for j, x in enumerate(label_loc):
            ax.scatter((x + center_bar[i]) + rand_m1(conc[:, j].size) * w/4,
                       conc[:, j], color=colors[i], edgecolor='k')

    ### Plot Stats significance if provided.
    if stats_list is not None:
        for i, loc in enumerate(label_loc):
            if stats_list[i] < 0.05:
                y = max(std_conc[0][i] + avg_conc[0][i], std_conc[1][i] + avg_conc[1][i]) + 4
                plt.text(loc, y, "*", ha="center", va="bottom", fontsize=30)     
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(label_loc)
    ax.set_xticklabels(x_labels)
    ax.legend()
    plt.legend()       
    ### Saving the figure with a timestamp as a name.
    if save_name is not None:
        plt.savefig(save_name)
