import numpy as np
import seaborn as sns
from matplotlib.patches import Circle


def heatmap(results, title=None, xlabel=None, ylabel=None, xticklabels="auto", yticklabels="auto", vmin=None, vmax=None,
            maxannotate=False, cbar=True):
    ax = sns.heatmap(results, linewidth=0.5, vmin=vmin, vmax=vmax, xticklabels=xticklabels, yticklabels=yticklabels,
                     cbar=cbar)
    ax.tick_params(labelrotation=0)

    if title:
        ax.set_title(title, fontsize=18)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=15)

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=15)

    if maxannotate:
        arr = np.array(results)
        for i, row in enumerate(arr):
            # ax.add_patch(Rectangle((row.argmax() + 0.4, i + 0.5), 0.1, 0.1, fill=True, color='red', edgecolor='red', lw=3))
            ax.add_patch(Circle((row.argmax() + 0.53, i + 0.53), 0.1, fill=True, color='red', edgecolor='red'))

    return ax
