"""
Small module with functions extending pairplots, etc.

@author:frbourassa
January 17, 2021
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from scipy.stats import gaussian_kde

def _add_custom_kde(data, var, lvl, idx, clrs, ax):
    # First, get two KDEs

    kde1 = gaussian_kde(data.loc[idx, var])
    kde2 = gaussian_kde(data.loc[np.logical_not(idx), var])
    xrange = np.linspace(*ax.get_xlim(), 151)
    y1 = kde1(xrange)
    y2 = kde2(xrange)
    # Second, rescale what we want to plot
    yscale = ax.get_ylim()[1]
    kdemax = max(np.amax(y1), np.amax(y2))
    y1 *= yscale / kdemax * 0.8
    y2 *= yscale / kdemax * 0.8
    ax.plot(xrange, y1, color=clrs[0], lw=1.)
    ax.plot(xrange, y2, color=clrs[1], lw=1.)
    ax.fill_between(xrange, y1, label=lvl[0], alpha=0.3, color=clrs[0])
    ax.fill_between(xrange, y2, label=lvl[1], alpha=0.3, color=clrs[1])
    return y1, y2


def dual_pairplot(data, vari, dual_lvl, dual_labels, dual_hues=None, **kwargs):
    """Produce a custom pairplot from scratch. Can't rely on seaborn pairplots and
    combine them because moving axes between figures causes problems when saving to pdf
    and is strongly discouraged.
    kwargs are passed to sns.scatterplot (for different hues, styles, sizes, etc.).
    One scatterplot legend is produced.
    """
    # Extract data
    where_1 = data[dual_lvl] == dual_labels[0]
    if dual_hues is None:
        dual_hues = plt.cm.viridis([40, 206])

    # Create figure and gridspec
    n = len(vari)
    fig = plt.figure()
    gs = fig.add_gridspec(n, n)
    axes = np.zeros([n, n], dtype=mpl.axes.Axes)

    # Bottom half of the plot: where_1 data, other half, ~where_1
    for i in range(n):
        for j in range(0, i):
            # Lower half of plot
            # Create axis, share x and y if possible
            sharey = None if (i <= 1 or j == 0) else axes[i, 0]
            sharex = None if (j == i-1) else axes[j+1, j]
            do_leg = "auto" if (i == 1 and j == 0) else False
            axes[i, j] = fig.add_subplot(gs[i, j], sharex=sharex, sharey=sharey)
            sns.scatterplot(data=data.loc[where_1], x=vari[j], y=vari[i], ax=axes[i, j], legend=do_leg, **kwargs)
            axes[i, j].text(0.5, 0.9, dual_labels[0], ha="center", transform=axes[i, j].transAxes)
            if do_leg == "auto":
                axes[i, j].get_legend().remove()
            # Upper half of plot
            axes[j, i] = fig.add_subplot(gs[j, i])
            axes[j, i].set_xlim(*axes[i, j].get_xlim())
            axes[j, i].set_ylim(*axes[i, j].get_ylim())
            sns.scatterplot(data=data.loc[~where_1], x=vari[j], y=vari[i], ax=axes[j, i], legend=False, **kwargs)
            axes[j, i].text(0.5, 0.9, dual_labels[1], ha="center", transform=axes[j, i].transAxes)
            # For upper half, mark the break in shared axes with thicker spines
            if j == i-1:
                for a in ["left", "bottom"]:
                    axes[j, i].spines[a].set_linewidth(3*axes[j, i].spines[a].get_linewidth())

    # Diagonal
    for i in range(n):
        if i < n-1:
            sharex = axes[n-1, i]
        else:
            sharex = None
        axes[i, i] = fig.add_subplot(gs[i, i], sharex=sharex)
        if sharex is None:  # Need to set proper limits if we can't share them
            axes[i, i].set_xlim(*axes[i, 0].get_ylim())
        _add_custom_kde(data, vari[i], dual_labels, where_1, dual_hues, ax=axes[i, i])
        axes[i, i].set_xlabel(vari[i])
        axes[i, i].set_ylabel("Density [-]")

    # Add one legend for the scatterplots
    dual_handles, dual_labels = axes[n-1, n-1].get_legend_handles_labels()
    scatter_handles, scatter_labels = axes[1, 0].get_legend_handles_labels()
    all_handles = scatter_handles + [mpl.patches.Patch(facecolor=(0, 0, 0, 0), edgecolor=(0, 0, 0, 0))] + dual_handles
    all_labels = scatter_labels + [dual_lvl] + dual_labels
    leg = fig.legend(handles=all_handles, labels=all_labels,
        frameon=True, loc="upper left", bbox_to_anchor=(0.07, 0.01), ncol=n*2,
        handlelength=1.5, handletextpad=0.5, borderpad=0.5)

    return fig, axes, leg
