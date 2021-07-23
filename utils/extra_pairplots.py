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

# Found on Github: https://gist.github.com/salotz/8b4542d7fe9ea3e2eacc1a2eef2532c5
def _move_axes(ax, fig, subplot_spec=(1, 1, 1)):
    """Move an Axes object from a figure to a new pyplot managed Figure in
    the specified subplot."""

    # get a reference to the old figure context so we can release it
    old_fig = ax.figure

    # remove the Axes from its original Figure context
    ax.remove()

    # set the pointer from the Axes to the new figure
    ax.figure = fig

    # add the Axes to the registry of axes for the figure
    fig.axes.append(ax)
    # twice, I don't know why...
    fig.add_axes(ax)

    # then to actually show the Axes in the new figure we have to make
    # a subplot with the positions etc for the Axes to go, so make a
    # subplot which will have a dummy Axes
    dummy_ax = fig.add_subplot(*subplot_spec)

    # then copy the relevant data from the dummy to the ax
    ax.set_position(dummy_ax.get_position())

    # then remove the dummy
    dummy_ax.remove()
    return ax

def _clear_axis(ax):
    for artist in ax.lines + ax.collections:
        artist.remove()

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


## WARNING: THIS FAILS WHEN SAVING TO VECTOR FORMATS,
## BECAUSE TRANSFORMS OF THE MOVED AXES ARE NOT PROPERLY SET
def dual_pairplot_hackish(data, vari, dual_lvl, dual_labels, dual_hues=None, **kwargs):
    """Combine two pairplots for the two labels dual_labels in level dual_lvl,
    one on the lower triangular part, the other on the upper triangular.

    Slow: calls pairplot twice, puts axes of the 2nd pairplot on the first one.

    All kwargs go directly to the pairplots.
    """
    where_1 = data[dual_lvl] == dual_labels[0]
    if dual_hues is None:
        dual_hues = plt.cm.viridis([40, 206])

    # Make the two pairplots. For the first, make only lower part;
    # For the second, we need to make both parts so we can extract
    # the upper triangular part and place it in the first plot.
    kwargs["diag_kind"] = None
    kwargs["corner"] = True
    pairplot1 = sns.pairplot(data.loc[where_1], vars=vari, **kwargs)
    kwargs["corner"] = False
    pairplot2 = sns.pairplot(data.loc[np.logical_not(where_1)], vars=vari, **kwargs)

    # Show all ticks and labels to make space for what is coming.
    for i in range(len(vari)):
        for j in range(0, i):
            for ax in [pairplot1.axes, pairplot2.axes]:
                ax[i, j].tick_params(labelleft=True, labelbottom=True)
                ax[i, j].set_ylabel(vari[i], visible=True)
                ax[i, j].set_xlabel(vari[j], visible=True)

    pairplot1.fig.tight_layout(h_pad=0.5, w_pad=0.5)
    pairplot2.fig.tight_layout(h_pad=0.5, w_pad=0.5)

    n = len(vari)
    for i in range(n):
        # Remove data on the diagonal
        _clear_axis(pairplot1.axes[i, i])
        # Plot instead a KDE of Synthetic vs Data
        _add_custom_kde(data, vari[i], dual_labels, where_1, dual_hues, ax=pairplot1.axes[i, i])
        for j in range(i+1, n):
            # Use the equivalent plot, not the mirror reflection, for easier comparison
            _move_axes(pairplot2.axes[j, i], pairplot1.fig, subplot_spec=(n, n, i*n + j+1))
            pairplot1.axes[i, j] = pairplot2.axes[j, i]
            ax2, ax1 = pairplot1.axes[i, j], pairplot1.axes[j, i]
            ax2.text(0.5, 0.9, dual_labels[1], ha="center", transform=ax2.transAxes)
            ax1.text(0.5, 0.9, dual_labels[0], ha="center", transform=ax1.transAxes)
            # For ax2, label axes again if we are just above the diagonal
            # and mark the break in shared axes with thicker spines
            if j == i+1:
                #ax2.set_xlabel(vari[j-1])
                #ax2.set_ylabel(vari[j])
                for a in ["left", "bottom"]:
                    ax2.spines[a].set_linewidth(3*ax2.spines[a].get_linewidth())

    # Add a legend for the hues on the diagonal.
    leg = pairplot1.fig.legend(
        *pairplot1.axes[n-1, n-1].get_legend_handles_labels(),
        frameon=False, loc="upper left", bbox_to_anchor=(1, 1))

    # close the figure the original axis was bound to
    plt.close(pairplot2.fig)

    return pairplot1, leg

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
