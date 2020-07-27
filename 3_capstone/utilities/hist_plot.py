
# Standard Library Imports

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def hist_plot(data, num_bins, labels, xy_max_min):
    # INPUT: data = input series, num_bins = data segements
    # OUTPUT :  plot the data

    import statistics
    mu    = statistics.mean(data)
    sigma = statistics.stdev(data)


    ymin = xy_max_min['ymin']
    ymax = xy_max_min['ymax']
    xmin = xy_max_min['xmin']
    xmax = xy_max_min['xmax']

    ylabel = labels['ylabel']
    xlabel = labels['xlabel']
    title  = labels['title']


    chart,axes = plt.subplots(1,1)
    chart.set_figheight(8)
    chart.set_figwidth(16)

    axes.hist(data, num_bins, color='olive', alpha =0.75, edgecolor='k',
              linestyle='dashed', linewidth=2, density=False)
    axes.set_xlim(xmin, xmax)
    axes.set_ylim(ymin, ymax)
    axes.set_ylabel(ylabel,fontsize=24)
    axes.set_xlabel(xlabel,fontsize=24)
    axes.set_title(title,fontsize=32, fontweight='bold')
    return
