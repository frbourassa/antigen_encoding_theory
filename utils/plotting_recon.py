""" Functions to generate supplementary figures related to cytokine
trajectory reconstruction.
@author:frbourassa
Jan 27, 2021
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from utils.plotting_fits import (read_conc, LegendSubtitle,
            LegendSubtitleHandler, add_hue_size_style_legend)
# For the KDE plots
from ltspcyt.scripts.reconstruction import ScalerKernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity


## Backup palette
palette_backup = plt.rcParams['axes.prop_cycle'].by_key()['color']
peptides_backup = ["N4", "Q4", "T4", "V4", "G4", "E1", "A2", "Y3", "A8", "Q7"]
colors_backup = {peptides_backup[i]:palette_backup[i] for i in range(len(peptides_backup))}

# Custom legend function
def add_recon_legend(fig, orig="-", recon="--",
                        sizes={}, sz_title="Concentration", **kwargs):
    """ Small function to add a legend telling apart reconstructed and
    original data as well as the different sizes.

    Args:
        fig (plt.figure.Figure): the figure on which to add the legend
        orig (str): line style for original data
        recon (str): line style for reconstruction
        sizes (dict): each key is a concentration/TCellNumber, and the
            associated value is the line width
        sz_title (str): title of the legend section for line widths.
        kwargs (dict): other keyword args are passed directly to axes.legend()
    Returns:
        leg (plt.legend.Legend): the legend.
    """
    # Reconstruction type part
    handles = []
    handles.append(Line2D([0], [0], color="k", ls=orig, label="Original"))
    handles.append(Line2D([0], [0], color="k", ls=recon, label="Recon."))

    # Sizes for different concentrations (or T Cell Number) part
    handles.append(LegendSubtitle(sz_title))
    for lbl in sorted(sizes.keys(), key=read_conc, reverse=True):
        handles.append(
            Line2D([0], [0], label=lbl, ls="-", lw=sizes[lbl], color="k"))

    # Put things together, add the legend
    # Add a column of axes to the right of the plot for the legend
    # so it may run over multiple rows.
    #axleg = fig.add_subplot(fig.axes.shape[0], fig.axes.shape[1]+1,
    #                        fig.axes.shape[1]+1)
    leg = fig.legend(handles=handles, fontsize=7,
            handler_map={LegendSubtitle: LegendSubtitleHandler()},
            loc="upper left", bbox_to_anchor=(0.98, 0.9), **kwargs)
    return leg

## Plotting function to compare reconstruction and data,
## or fit and data (latent space)
# Function to plot a comparison of the reconstruction and the original data, for a given feature
def plot_recon_true(df_full, df_r_full, feature="integral", toplevel="Data",
    sharey=True, do_legend=True, colors=colors_backup, pept=peptides_backup):
    """
    Args:
        df_full (pd.DataFrame): the dataframe for the experimental data
        df_r (pd.DataFrame): the reconstructed data
        feature (str): the feature to compare ("integral", "concentration", "derivative")
        toplevel (str): the first index level, one plot per entry
        sharey (bool): whether or not the y axis on each row should be shared
            True by default, allows to see if somne cytokines weigh less in the reconstruction.
        do_legend (bool): if True, add a legend for line styles
        colors (dict): dict of colors, at least as long as pept,
            keys are elements of pept
        pept (list): list of peptides
    """
    # Slice for the desired feature
    df = df_full.xs(feature, level="Feature", axis=1, drop_level=True)
    df_r = df_r_full.xs(feature, level="Feature", axis=1, drop_level=True)

    # Plot the result
    # Rows are for peptides, columns for cytokines
    # One panel per dataset
    figlist = {}
    for xp in df.index.get_level_values(toplevel).unique():
        # Extract labels
        cols = df.loc[xp].index.get_level_values("Peptide").unique()
        cols = [p for p in pept if p in cols]  # Use the right order
        try:
            rows = df.loc[xp].columns.get_level_values("Cytokine").unique()
        except KeyError:
            rows = df.loc[xp].columns.get_level_values("Node").unique()
            print("Reconstructing latent space")
        concs_num = map(read_conc, df.index.get_level_values("Concentration").unique())
        concs = np.asarray(df.index.get_level_values("Concentration").unique())
        concs = concs[np.argsort(list(concs_num))]
        sizes = {concs[i]:1.5 + 0.5*i for i in range(len(concs))}
        nice_cytos = {"IFNg":r"IFN-$\gamma$", "TNFa":"TNF"}

        # Make sure axes is a 2D array
        fig, axes = plt.subplots(len(rows), len(cols), sharex=True, sharey=sharey)
        if len(rows) == 1:
            axes = np.asarray([axes])
        elif len(cols) == 1:
            axes = axes[:, None]
        fig.set_size_inches(6.5, 1.1*len(rows))
        times = df.loc[xp].index.get_level_values("Time").unique()
        times = [float(t) for t in times]
        for i, cyt in enumerate(rows):
            for j, pep in enumerate(cols):
                for k in concs:
                    try:
                        li1, = axes[i, j].plot(times, df.loc[(xp, pep, k), cyt],
                                    color=colors[pep], lw=sizes[k], ls="-")
                        li2, = axes[i, j].plot(times, df_r.loc[(xp, pep, k), cyt],
                                    color=colors[pep], lw=sizes[k], ls="--")
                    except KeyError:  # This combination dos not exist
                        continue
                # Some labeling
                if j == 0:
                    axes[i, j].set_ylabel(
                        r"$\log_{10}$" + nice_cytos.get(cyt, cyt), size=8)
                if i == len(rows) - 1:
                    axes[i, j].set_xlabel("Time (h)", size=8)
                if i == 0:
                    axes[i, j].set_title(pep, size=8)
                axes[i, j].tick_params(axis="both", length=2.5, width=0.8, labelsize=7)
        # Add a legend
        leg = add_recon_legend(fig, orig="-", recon="--",
            sizes=sizes, sz_title="Concentration", frameon=False)
        leg.get_frame().set_facecolor('none')

        # Save the figure afterwards
        fig.tight_layout()
        figlist[xp] = fig
    return figlist


def plot_residuals_summary(df_res_list, df_names, feat="concentration",
            sharey_all=True, legend_loc="side"):
    """ Prepare side-by-side plots, showing a summary of residuals over time for ncols
    different reconstruction methods (df_res_list), one row for each cytokine

    legend_loc == "side" or "bottom"
    """
    # Prepare the plot grid.
    cytokines = df_res_list[0].columns.get_level_values("Cytokine").unique()
    nice_cytos = {"IFNg":r"IFN-$\gamma$", "TNFa":"TNF"}
    nrows = len(cytokines)
    ncols = len(df_res_list)
    fig = plt.figure()
    if legend_loc == "side":
        gs = fig.add_gridspec(nrows, ncols + 1)
    elif legend_loc == "bottom":
        gs = fig.add_gridspec(nrows, ncols)
    else:
        raise ValueError("Accepted legend_loc are 'side', 'bottom'")
    # Last column is for the legend.

    # Compute the statistics to plot
    # Compute relative to the max for each cytokine.
    gpbys = [df.groupby(["Time"]) for df in df_res_list]
    mins = [g.min() for g in gpbys]
    maxs = [g.max() for g in gpbys]
    means = [g.mean() for g in gpbys]
    stds = [g.std() for g in gpbys]
    times_lbls = mins[0].index.get_level_values("Time").unique()
    times = times_lbls.astype(float).sort_values()
    tim_argsort = np.argsort(times)
    times = times[tim_argsort]
    times_lbls = times_lbls[tim_argsort]


    # Plot nicely those statistics around the mean, as a function of time
    cmap = sns.color_palette("Greys", 4)[::-1]
    axes = [[None]*ncols]*nrows

    # We share y across all cytokines, to show how some cytokines are more accurately reconstructed.
    for i in range(nrows):
        cyt = cytokines[i]
        for k in range(ncols):
            # Create axis, share x and/or y axes properly
            if k > 0 and i > 0:  # Can share both
                ax = fig.add_subplot(gs[i, k], sharex=axes[0][k], sharey=axes[0][0])
            elif k > 0:  # first row still, can share y
                ax = fig.add_subplot(gs[i, k], sharey=axes[0][0])
            elif i > 0:  # First column, can share x
                ax = fig.add_subplot(gs[i, k], sharex=axes[0][k])
            else:  # First plot
                ax = fig.add_subplot(gs[i, k])
            axes[i][k] = ax

            # Hiding ticks, labeling axes, etc.
            ax.tick_params(axis="both", labelsize=7, length=2., width=0.8)
            if i != nrows - 1:
                for lbl in ax.xaxis.get_ticklabels():
                    lbl.set_visible(False)
            else:
                ax.set_xlabel("Time (h)", size=8)

            if k > 0:
                for lbl in ax.yaxis.get_ticklabels():
                    lbl.set_visible(False)
            else:
                ax.set_ylabel(
                    r"$\Delta \, \log_{10}$ " + nice_cytos.get(cyt, cyt), size=8)
                if sharey_all:
                    ax.set_ylim((
                        min([g.min().min() for g in mins]),
                        max([g.max().max() for g in maxs])
                    ))

            # Title of each column
            if i == 0:
                ax.set_title(df_names[k], size=8)

            # Plotting the residuals' statistics
            mn = means[k].loc[times_lbls, cyt]
            st = stds[k].loc[times_lbls, cyt]
            # Fill between the \pm standard deviation
            ax.fill_between(times, mn-st, mn+st, color=cmap[1], alpha=0.5)
            # Then plot the rest on top of that.
            ax.axhline(0, lw=2., color="xkcd:magenta", ls="-.")
            ax.plot(times, mn, label="Mean", color=cmap[0], lw=3.)
            ax.plot(times, mn + st, color=cmap[1], ls="--", lw=2.2, label="Std dev.", )
            ax.plot(times, mn - st, color=cmap[1], ls="--", lw=2.2)
            ax.plot(times, mins[k].loc[times_lbls, cyt], color=cmap[2], ls=":", lw=1.4, label="Extrema")
            ax.plot(times, maxs[k].loc[times_lbls, cyt], color=cmap[2], ls=":", lw=1.4)

    # Add legend
    if legend_loc == "side":
        axleg = fig.add_subplot(gs[:, -1])
        axleg.set_axis_off()
    elif legend_loc == "bottom":
        axleg = fig
        fig.subplots_adjust(bottom=0.2)
    else: pass  # we raised error above

    handles, labels = axes[0][0].get_legend_handles_labels()
    #for hd in handles:
        #hd.set_color("k")
    if legend_loc == "side":
        leg_args = dict(
            fontsize=8, bbox_to_anchor=(0, 0.97), loc="upper left", ncol=1,
            frameon=False, borderaxespad=-1.0, borderpad=0.0)
    elif legend_loc == "bottom":
        leg_args = dict(
            fontsize=8, bbox_to_anchor=(0.15, 0), loc="lower left", ncol=2,
            frameon=True, borderaxespad=0.0, borderpad=0.5)

    axleg.legend(handles, labels, **leg_args)
    if legend_loc == "side":
        fig.set_size_inches(3.5, nrows*1.1)
    else:
        fig.set_size_inches(3., nrows*1.1)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.5, w_pad=0.5)
    plt.show()
    plt.close()
    return fig, axes, axleg


## Scoring/performance evaluation functions
# Function to compute some performance metrics
def performance_recon(df_full, df_r_full, feature="integral", toplevel=None):
    """
    Args:
        df_full (pd.DataFrame): the dataframe for the experimental data
        df_r (pd.DataFrame): the reconstructed data
        feature (str): the feature to compare ("integral", "concentration", "derivative")
        toplevel (str): the first index level, one plot per entry
    Returns:
        res_per_point
        histogs
        bins
        r2
    """
    # Slice for the desired feature
    df = df_full.xs(feature, level="Feature", axis=1, drop_level=True)
    df_r = df_r_full.xs(feature, level="Feature", axis=1, drop_level=True)

    # Residual vectors
    residuals = df - df_r
    # Squared norm of each residual vector
    residuals_norms = (residuals**2).sum(axis=1)

    # If toplevel is specified, do this per entry
    # If not, only one pass in the loop.
    if toplevel is None:
        totalres = residuals_norms.sum()
        nb = residuals_norms.shape[0]
        # histogram of the residuals, per dimension
        ndims = len(residuals.columns)
        histogs = pd.Series([[]]*ndims, index=residuals.columns)
        bins = pd.Series([[]]*ndims, index=residuals.columns)
        for dim in residuals.columns:
            histogs[dim], bins[dim] = np.histogram(residuals[dim])
    else:
        # Total residuals
        totalres = residuals_norms.groupby(toplevel).sum()
        # Divided by number of points in each group
        nb = residuals_norms.groupby(toplevel).count()
        # histogram of the residuals, per dimension and per dataset
        ndims = len(residuals.columns)
        topindex = residuals.index.get_level_values(toplevel).unique()
        nind = len(topindex)
        histogs = pd.DataFrame([[[]]*ndims]*nind, columns=residuals.columns, index=topindex)
        bins = histogs.copy()
        for dim in residuals.columns:
            for key in topindex:
                histogs.at[key, dim], bins.at[key, dim] = np.histogram(residuals.loc[key, dim])

    # In both cases, res_per_point is defined the same way.
    res_per_point = np.divide(totalres, nb)

    # R2 coefficient, for comparison with sklearn
    r2 = 1 - residuals_norms.sum()/((df - df.mean(axis=0))**2).sum().sum()

    return res_per_point, histogs, bins, r2


def plot_histograms(df_points, df_bins, xlabel="Residuals"):
    try:
        cols = df_points.columns
    except AttributeError:  # just a Series
        df_points = pd.DataFrame(df_points, index=[0], columns=df_points.index)
        df_bins = pd.DataFrame(df_bins, index=[0], columns=df_bins.index)
        cols = df_points.columns

    rows = df_points.index
    fig, axes = plt.subplots(len(rows), len(cols), sharey=True)  # can share y, same nb of points in each component
    if len(rows) == 1:
        axes = axes.reshape(1, len(cols))  # Make sure it's 2D
    fig.set_size_inches(1.5*len(cols), 1.5*len(rows))  # small histograms indeed
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            axes[i, j].hist(df_bins.loc[r, c][:-1], bins=df_bins.loc[r, c], weights=df_points.loc[r, c])
    for ax in axes[-1, :]:
        ax.set_xlabel(xlabel)
    return fig, axes


def pairplot_scalerkdes(kde_dict, hue_order, vari_to_plot, do_leg=True,
                        hues={}, res=51, plot_type="line", **kwargs):
    """ Create small parplots of KDEs in parameter space.

    Args:

    """
    # Find out the number of dimensions of the KDEs,
    # check whether it matches the variable names given.
    ndimensions = list(kde_dict.values())[0].sample(1).size
    n_par = len(vari_to_plot)
    assert ndimensions >= n_par, \
        "Excessive number of parameter names given"
    assert set(kde_dict.keys()) == set(hue_order), \
        "kde_dict contains unknown colors"

    if hues == {}:
        palette = sns.color_palette(n_colors=len(kde_dict))
        hues = {k:palette[i] for i, k in enumerate(hue_order)}
    else:
        assert kde_dict.keys() == hues.keys()

    # Map for nicer label formatting
    nice_param_names = {"a0":"a_0", "theta":r"\theta", "v1":"v_{t1}", "t0":"t_0",
                        "alpha":r"\alpha", "beta":r"\beta", "tau0":r"\tau_0"}

    # Prepare the plot's axes, lower triangular layout
    # Use the upper triangular part for the legend
    # Loop over pairs of parameters, column-major, so the first column
    # parameter gets the most plots (first column, nparams-1 rows)
    fig = plt.figure()
    gs = fig.add_gridspec(n_par - 1, n_par - 1)
    axes = [[None for i in range(n_par - 1)] for j in range(n_par - 1)]
    param_pairs = []
    idx_pairs = []
    for j in range(n_par-1):
        for i in range(j, n_par-1):
            # Pair of parameters corresponding to that subplot
            param_pairs.append((vari_to_plot[j], vari_to_plot[i+1]))
            idx_pairs.append((i, j))
            axes[i][j] = fig.add_subplot(gs[i, j])

    # Combine grid cells to make space for the legend, as much as possible
    # Number of rows in the upper triangular, diagonal excluded = n_par - 2
    if n_par - 2 == 0:
        # Case where the anti-diagonal has one upper element, so 1*2 > 1*1
        axleg = fig.add_subplot(1, 2, 2)  # Use all the last column
    elif (n_par - 2) % 2 == 1:
        # Use the antidiagonal square
        antid_size = (n_par - 2) // 2 + 1
        axleg = fig.add_subplot(gs[:antid_size, -antid_size:])
    else:
        # Use the antidiagonal square plus one row below
        antid_size = (n_par - 2) // 2
        axleg = fig.add_subplot(gs[:antid_size+1, -antid_size:])


    # Loop over pairs and over each hue group
    for ii, pair in enumerate(param_pairs):
        i, j = idx_pairs[ii]
        for key, skde in kde_dict.items():
            # Fit a marginal ScalerKernelDensity
            #data_2d = np.asarray(skde.kde.tree_.data)
            #data_2d = skde.scaler.inverse_transform(data_2d)
            data_2d = skde.sample(1000)
            data_2d = data_2d[:, [j, i+1]]
            scaler = StandardScaler().fit(data_2d)
            kde_marg = ScalerKernelDensity(scaler=scaler, kernel="gaussian",
                            bandwidth=skde.kde.bandwidth).fit(data_2d)

            # Compute the ScalerKernelDensity on a grid
            lower_lims = scaler.transform(data_2d.min(axis=0)[np.newaxis, :]) - 0.3*kde_marg.kde.bandwidth
            upper_lims = scaler.transform(data_2d.max(axis=0)[np.newaxis, :]) + 0.3*kde_marg.kde.bandwidth
            lower_lims = scaler.inverse_transform(lower_lims)
            upper_lims = scaler.inverse_transform(upper_lims)
            xx, yy = np.meshgrid(
                np.linspace(lower_lims[0, 0], upper_lims[0, 0], res),
                np.linspace(lower_lims[0, 1], upper_lims[0, 1], res))
            xy_list = np.stack([xx.flatten(), yy.flatten()], axis=1)
            probdens = np.exp(kde_marg.score_samples(xy_list)).reshape(xx.shape)

            # Contour plot
            mndens = np.mean(probdens)
            maxdens = np.max(probdens)
            if plot_type == "fill":
                palet = [list(hues[key])+[0.4], list(hues[key]) + [0.7]]
                axes[i][j].contourf(xx, yy, probdens, colors=palet,
                    levels=[(3*mndens + maxdens)/4, (mndens + 2*maxdens)/3, maxdens],
                    linewidths=[0.5, 0.8], zorder=len(hue_order) - hue_order.index(key))
            else:
                axes[i][j].contour(xx, yy, probdens, colors=list(hues[key]) + [0.7],
                    levels=[(3*mndens + maxdens)/4, (mndens + 2*maxdens)/3, maxdens],
                    linewidths=[0.5, 0.8], zorder=len(hue_order) - hue_order.index(key))

        # Label this plot
        if i > j and j > 0:
            axes[i][j].set_xlim(*axes[j][j].get_xlim())
            axes[i][j].set_ylim(*axes[i][0].get_ylim())
        elif i > j:
            axes[i][j].set_xlim(*axes[j][j].get_xlim())
        elif j > 0:
            axes[i][j].set_ylim(*axes[i][0].get_ylim())

        if i == n_par - 2:
            axes[i][j].set_xlabel(r"${}$".format(nice_param_names.get(pair[0], pair[0])),
                                  size=8, labelpad=0.8)
        else:
            axes[i][j].set_xticklabels([])
        if j == 0:
            axes[i][j].set_ylabel(r"${}$".format(nice_param_names.get(pair[1], pair[1])),
                                  size=8, labelpad=0.8)
        else:
            axes[i][j].set_yticklabels([])
        axes[i][j].tick_params(which="both", length=2., width=0.5, labelsize=6.)

    # Add a legend, etc.
    if do_leg:
        axleg.set_axis_off()
        if "ncol" not in kwargs.keys():
            kwargs["ncol"] = (n_par - 2)//2
        leg = add_hue_size_style_legend(
            axleg, hues, {}, {}, "", "",
            hue_sort_key=None, **kwargs)

    else:
        leg = None
        axleg = None
    return [fig, [axes, axleg, leg]]


if __name__ == "__main__":
    tests()
