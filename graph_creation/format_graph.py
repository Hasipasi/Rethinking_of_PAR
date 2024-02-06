from matplotlib import pyplot as plt
import numpy as np
def format_graph(metric, num_epochs, ymin=0.8, ymax=1.0):
    """
    Format the graph with labels, limits, ticks, and gridlines.
    """
    plt.ylabel(f'{metric}')
    plt.xlabel('Epoch num')
    plt.legend(loc='upper left') 
    plt.ylim(ymin, ymax)
    plt.xlim(0, 19)
    xticklabels = np.arange(1, num_epochs+1, step=1)
    plt.xticks(range(num_epochs), xticklabels)
    plt.grid(linewidth=1, linestyle='-', alpha=1)
    plt.minorticks_on()  # Enable minor ticks
    plt.grid(True, which='minor', linestyle='--', linewidth=0.5)  # Add minor gridlines