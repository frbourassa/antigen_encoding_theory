# -*- coding:utf-8 -*-
"""
Script to import the raw data, smooth it lightly with a moving average,
fit cubic splines to it, and save the cubic splines in a DataFrame.

Search "TODO" to see where to enter the file name (in the main)
and, if desired, the processing parameters (in make_spline_dataframe).


@author:frbourassa
August 13, 2019
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib.legend_handler import HandlerBase
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
import pickle
import os
import seaborn as sns

import os, sys
main_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, main_dir_path)

# To extract information from dataframes with variable formats.
from utils.plotting_splines import (nicer_name, index_split,
                                find_peptide_concentration_names, process_file)
# To add subtitles in legends
from utils.plotting_fits import LegendSubtitle,LegendSubtitleHandler,read_conc


# Plot parameters
sns.reset_orig()
sns.set_palette("colorblind", 8)
plt.rcParams["figure.figsize"] = 4.75, 6
plt.rcParams["lines.linewidth"] = 0.8
plt.rcParams["font.size"] = 7.
plt.rcParams["axes.labelsize"] = 7.
plt.rcParams["legend.fontsize"] = 6.
plt.rcParams["xtick.labelsize"] = 6.
plt.rcParams["ytick.labelsize"] = 6.

# Scaling the axes lines themselves.
plt.rcParams["xtick.major.size"] = 2.
plt.rcParams["ytick.major.size"] = 2.
plt.rcParams["xtick.major.width"] = 0.6
plt.rcParams["ytick.major.width"] = 0.6
plt.rcParams['axes.linewidth'] = 0.6

plt.rcParams['savefig.transparent'] = True


def make_spline_dataframe(filename, do_save=False):
    """ Import the pickled cytokine data located at filename and
    return and save (in the output/splines folder) a DataFrame containing the
    sp.interpolate.UnivariateSpline objects.

    Args:
        filename (str): path and file name, including extension.
        do_save (bool): if True, the spline dataframe is pickled.

    Returns:
        spline_frame (pd.DataFrame): a DataFrame with the conditions (Peptide,
            Concentration, TCellType, etc.) in the index and "Cytokine"
            in the columns. There is one spline per condition, per cytokine.
        data (pd.DataFrame): the raw data, formatted like the spline_frame,
            but with Time too in the columns (because there is one entry per
            time point here, compared to one spline object covering all times)
        data_log (pd.DataFrame): the raw data in log scale and shifted so
            the minimum value is 1 (chose the minimum value to be 1 to prevent
            having negative values due to cubic splines overshooting)
        data_smooth (pd.DataFrame): the log data smoothed with a moving average
    """
    # TODO: choose processing arguments
    processing_args = dict(
        take_log = True,
        rescale_max = False,
        smooth_size = 3,  # for the moving average
        rtol_splines = 1/5,  # fraction of residuals between raw and moving avg
        keep_mutants = True,
        tcelltype = None, # all types will be loaded if None
        antibody = None,
        genotype = None,
        lod_folder = os.path.join("data", "LOD/")
    )

    # We consider only the 6 most responsive cytokines
    good_cytokines = ["IFNg", "IL-2", "IL-17A", "IL-6", "TNFa"]

    # Import and process the data
    fullpath = os.path.split(filename)
    filepath = os.path.join(*fullpath[:-1])
    ret = process_file(filepath, fullpath[-1], **processing_args)

    # Return the data at different stages of processing
    # (first 4 returns of process_file)
    data, data_log, data_smooth, spline_frame = ret[0:4]
    spline_frame = spline_frame.loc[:, good_cytokines]
    data = data.loc[:, good_cytokines]
    data_log = data_log.loc[:, good_cytokines]
    data_smooth = data_smooth.loc[:, good_cytokines]

    # Save the splines to the output/splines/ folder
    if do_save:
        prefix = os.path.join("data", "splines", "spline_dataframe_")
        exp_name = nicer_name(filename)
        frame_file_name = prefix + exp_name + ".pkl"
        with open(frame_file_name, "wb") as handle:
            pickle.dump(spline_frame, handle)

    return spline_frame, data, data_log, data_smooth


def color_quality_lightness_quantity(labels, colorcycle=None):
    """A function that returns a list of colors corresponding
    to the series labels. Each peptide gets a different color,
    and each quantity gets a different intensity.

    Args:
        labels (list of tuples): each tuple is (peptide, concentration), with
            the concentration as a string, ending with units of uM, nM, etc.
        colorcycle (list of colors): the colors though which we cycle
            to assign one color to each peptide. If None (default),
            will use the default color cycle in plt.rcParams.
    """
    # If no colorcycle was given, get the default one.
    if colorcycle is None:
        colorcycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Associate a color to each peptide, a lightness to each concentration.
    # zip all peptides together, all conc together.
    pep_list, conc_list = list(zip(*labels))
    # keep one instance of each peptide, in the order they appear
    pep_list = pd.unique(np.array(pep_list))
    conc_list = set(conc_list)
    colors_dict = {}

    for pep in pep_list:
        next_color = colorcycle.pop(0)
        colors_dict[pep] = next_color
        colorcycle.append(next_color)

    # Normalize conc'ns between 0.15 (max) and 0.85 (min) logarithmically
    conc_vals = np.array([read_conc(a) for a in conc_list])
    # Before taking the log, replace 0 by 10 times less the nonzero minconc
    minconc, maxconc = np.amin(conc_vals), np.amax(conc_vals)

    if minconc != maxconc:
        if minconc == 0:
            nonzero_min = np.amin(conc_vals[conc_vals > 0])
            conc_vals[conc_vals == 0] = nonzero_min / 10
            minconc = nonzero_min / 10
        minconc, maxconc = np.log10(minconc), np.log10(maxconc)
        conc_vals = np.log10(conc_vals)
        # Want to map [minconc, maxconc] to [lmin, lmax]
        # with minconc to lmax, maxconc to lmin\
        lmin, lmax = 0.25, 0.85
        def scaler(x):
            return lmax - (lmax - lmin)/(maxconc - minconc) * (x - minconc)
    # Maybe there's only one concentration
    else:
        scaler = lambda x: None  # Use default hsl of that color.

    # To be used as hue, lightness or saturation
    light_dict ={lbl:scaler(conc) for lbl, conc in zip(conc_list, conc_vals)}

    # Finally, build a list of one color per label
    built_colors = []
    for lbl in labels:
        col = colors_dict[lbl[0]]
        brightness = light_dict[lbl[1]]
        # Use the seaborn function to modify the lightness (l) of color col
        built_colors.append(sns.set_hls_values(col, l=brightness))

    return built_colors


def add_processing_legend(ax, handles, example_raw, example_knot=None):
    """ Add a legend in the given ax or fig containing the existing handles (expected
        to be solid lines) and a section explaining the types of curves.
    """
    # Title
    handles.append(LegendSubtitle("Processing:"))

    # Curve shapes
    handles.append(Line2D([0], [0], color='grey',
                            label="Spline", linestyle="-"))
    raw_ls = example_raw.get_linestyle()
    raw_ms = example_raw.get_markersize()
    raw_mew = example_raw.get_markeredgewidth()
    raw_m = example_raw.get_marker()

    handles.append(Line2D([0], [0], marker=raw_m, color='grey',
                             label="Log data", linestyle=raw_ls,
                             markeredgecolor="grey", markersize=raw_ms,
                             markeredgewidth=raw_mew, markerfacecolor="grey"))
    if example_knot is not None:
        knt_m = example_knot.get_marker()
        knt_ls = example_knot.get_linestyle()
        knt_ms = example_knot.get_markersize()
        knt_mew = example_knot.get_markeredgewidth()
        knt_mec = example_knot.get_markeredgecolor()

        handles.append(Line2D([0], [0], marker=knt_m, color='grey',
                            label="Spline knots", linestyle=knt_ls,
                            markeredgecolor=knt_mec, markersize=knt_ms,
                            markeredgewidth=knt_mew, markerfacecolor="grey"))

    ax.legend(handles=handles,
              handler_map={LegendSubtitle: LegendSubtitleHandler()},
              loc="upper left", bbox_to_anchor=(1, 1), handlelength=1.25)


def plot_splines_vs_data(df_spline, df_log, df_smooth, pep_conc_names,
                            spl_times=None, do_knots=False):
    """ For each peptide at all concentrations, plot the cubic splines
    against the rescaled time series of each cytokine. Assumes that
    the index of the DataFrames only contains Peptide and Concentration.

    Args:
        df_spline (pd.DataFrame) : see compare_splines_data's doc
        df_log (pd.DataFrame) : see compare_splines_data's doc
        df_smooth (pd.DataFrame) : see compare_splines_data's doc
        pep_conc_names (list): level names for Peptide and Concentration
        spl_times (1darray): the time axis of the plots
        do_knots (bool): if True, show where the spline knots are.
    Returns:
        fig, axes
    """
    exp_timepoints = df_log.columns.get_level_values("Time").unique()
    if spl_times is None:
        spl_times = np.linspace(0, exp_timepoints[-1], 201)

    # Identify each column to a peptide
    pep_name, conc_name = pep_conc_names
    peptides = df_spline.index.get_level_values(pep_name)
    concentrations = df_spline.index.get_level_values(conc_name)
    column_assignment = {a:i for i, a in enumerate(peptides.unique())}
    nice_cyto_names = {"IFNg":r"IFN-$\gamma$", "IL-2":"IL-2", "IL-6":"IL-6", "TNFa":"TNF", "IL-17A":"IL-17A"}

    # One cytokine per row, one peptide per column
    nrows = len(df_spline.columns.get_level_values("Cytokine").unique())
    ncols = len(peptides.unique())

    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey="row")
    # Science: 2.25 or 4.75 in width when printed, so plan accordingly.
    fig.set_size_inches(min(4*(ncols + 1), 4.75), min(4*nrows, 5.5))

    # Generate colors for peptides and concentrations
    # Assume the index will have the right length
    colors = color_quality_lightness_quantity(list(df_spline.index))
    sr_colors = pd.Series(colors, index=df_spline.index)

    # Plot one cytokine at a time (one row at a time)
    for i, cyt in enumerate(df_spline.columns):
        # Reset the list of curves every cytokine so we have all colors once.
        legend_handles = []
        for j in range(len(peptides)):
            pep, conc = peptides[j], concentrations[j]
            lbl = df_spline.index[j]  # either (pep, conc) or (conc, pep)
            nice_lbl = pep + " [{}]".format(conc)
            col = column_assignment[pep]
            ax = axes[i, col]
            clr = sr_colors[lbl]
            spline = df_spline.loc[lbl, cyt]
            smoothcurve = df_smooth.loc[lbl, cyt]
            logcurve = df_log.loc[lbl, cyt]
            # Plot the 2 curves and save the solid curve for the legend
            li, = ax.plot(spl_times, spline(spl_times),
                color=clr, ls="-", label=nice_lbl)
            legend_handles.append(li)
            #ax.plot(exp_timepoints, df_smooth.loc[lbl, cyt], color=clr, ls="--",
                #marker="^", ms=1.75, mfc=clr, mec=clr)
            # Also plot the knots of that spline.
            rawli, = ax.plot(exp_timepoints, df_log.loc[lbl, cyt], color=clr,
                ls=":", marker="o", ms=1.75, mfc=clr, mec=clr)
            if do_knots:
                knots = spline.get_knots()
                knotli, = ax.plot(knots, spline(knots),
                    ls="none", marker="^", ms=2.25, mec="r", mfc=clr, mew=0.75)
            else:
                knotli = None
            # Force more ticks...
            if i == len(df_spline.columns) - 1:
                ax.set_xticks([0, 20, 40, 60])
                ax.set_xticklabels([0, 20, 40, 60])
                ax.set_xlabel("Time [h]")
            elif i == 0:
                ax.set_title(pep, size=8)
            if j == 0:
                ax.set_ylabel(r"$\log_{10}$ " + nice_cyto_names.get(cyt, cyt)
                                + "/LOD [-]", size=6)

    # After all the cytokines are done, add a nice categorical legend
    # (peptide, type of curves)
    legax = fig.add_subplot(1, 1, 1)
    legax.set_axis_off()
    # That does not change the figure size, fine.
    add_processing_legend(legax, legend_handles, rawli, knotli)

    return fig, axes


def compare_splines_data(df_spline, df_log, df_smooth, spl_times=None,
                        do_knots=False, showplots=False, nice_name="notitle"):
    """ For each peptide at all concentrations, plot the cubic splines
    against the rescaled time series of each cytokine. Produces one plot
    per TCellType, Genotype, TCellNumber, Antibody.

    Args:
        df_spline (pd.DataFrame): df containing the spline objects.
        df_log (pd.DataFrame): df containing the log+rescaled time series.
        df_smooth (pd.DataFrame): df containing the smoothed time series.
        spl_times (1darray): time axis of the plots.
        do_knots (bool): instead of comparing splines to data, plot
            splines with the position of their knots.
        showplots (bool): if True, each plot will be shown as it is produced.
            False by default.
        nice_name (str): the name of the experiment.

    Returns:
        figures (list of matplotlib.figure.Figure): list of the Figures
        axeslist (list np.2darray): list of arrays of matplotlib.axes.Axes
    """
    # Find the level names for Peptide and Concentration to remove from
    # the indexing of the different plot panels, i.e., the levels in to_remove
    # will show up in each panel of plots.
    to_remove = find_peptide_concentration_names(df_spline.index)
    index_entries, absent = index_split(df_spline.index.copy(), to_remove)

    # If Peptide or Concentration could not be removed, add levels to
    # the DataFrames so after splitting them, there are two levels left
    # to plot each panel
    dfs = [df_spline, df_log, df_smooth]
    if len(absent) == 1:
        if absent[0] == to_remove[0]:  # missing Peptide-like entry
            extra = {"Peptide":"Peptide"}
            to_remove[0] = "Peptide"
        elif absent[0] == to_remove[1]:  # missing Concentration-like
            extra = {"Concentration":"Concentration"}
            to_remove[1] = "Concentration"
        for i in range(len(dfs)):
            dfs[i].assign(**extra).set_index(extra.keys(), append=True)

    elif len(absent) == 2:
        lvls = {"Peptide":"Peptide", "Concentration":"Concentration"}
        for i in range(len(dfs)):
            dfs[i].assign(**lvls).set_index(extra.keys(), append=True)
        to_remove = ["Peptide", "Concentration"]

    # Reorder the DataFrames to make sure the levels to slice are outer.
    other_levels = list(df_spline.index.names)
    for lvl in to_remove:
        other_levels.remove(lvl)
    for i in range(len(dfs)):
        dfs[i] = dfs[i].reorder_levels(other_levels + to_remove, axis=0)
        #dfs[i] = dfs[i].sort_index()
    df_spline, df_log, df_smooth = dfs

    # If there are multiple TCellType, TCellNumber, etc. in addition to
    # Peptide, Concentration conditions, prepare a different plot for each.
    # Prepare a list of sub-dataframes with only Peptide, Concentration.
    list_spline_df, list_log_df, list_smooth_df = [], [], []
    if len(index_entries) == 0:
        list_spline_df = [df_spline]
        list_log_df = [df_log]
        list_smooth_df = [df_smooth]
        index_entries = [("WT",)]
    else:
        for lbl in index_entries:
            list_spline_df.append(df_spline.loc[lbl])
            list_log_df.append(df_log.loc[lbl])
            list_smooth_df.append(df_smooth.loc[lbl])

    # Prepare one plot per sub-frame. Call a sub-function for this.
    figures = []
    axeslist = []
    for i in range(len(index_entries)):
        dfs =  [list_spline_df[i], list_log_df[i], list_smooth_df[i]]
        # Prepare a title: conditions joined by a comma
        fig_title = nice_name + "_" + "_".join(map(str, index_entries[i]))
        fig_title = fig_title.replace("/", "")  # avoid fake folder changes

        print("Drawing a plot panel for", fig_title)
        fig, axes = plot_splines_vs_data(*dfs, to_remove, spl_times, do_knots)
        figures.append(fig)
        axeslist.append(axes)
        if showplots:
            fig.subplots_adjust(top=0.9)  # To prevent title overlap
            fig.suptitle(fig_title, size=8)
            plt.show()
            plt.close()
        else:
            # Reduce space between subplots, because we are in a small-figure regime
            fig.subplots_adjust(wspace=0.1, hspace=0.1)
            fig.tight_layout()
            figtype = "spline_knots_" if do_knots else "spline_comparison_"
            figpath = os.path.join("figures", "supp")
            # Create a folder for the dataset
            if not os.path.exists(figpath):
                os.makedirs(figpath)
            fig.savefig(os.path.join(figpath, figtype + fig_title + ".pdf"),
                        format="pdf", dpi=200, transparent=True)
            plt.close()

    return figures, axeslist


def knots_info(df_spline):
    """ Generate plots and a DataFrame with information about the cubic splines.
    """
    # Build two DataFrames containing, respectively,
    # the number of knots and the residuals of each spline
    df_knots = pd.DataFrame(np.zeros(df_spline.shape, dtype=int),
        index=df_spline.index, columns=df_spline.columns)
    df_resid = pd.DataFrame(np.zeros(df_spline.shape),
        index=df_spline.index, columns=df_spline.columns)
    for i in range(df_spline.shape[0]):
        for j in range(df_spline.shape[1]):
            df_knots.iat[i, j] = df_spline.iat[i, j].get_knots().size
            df_resid.iat[i, j] = df_spline.iat[i, j].get_residual()
    # Concatenate the two DataFrames into one, with another column level.
    labeled = {"NumberKnots":df_knots, "Residuals":df_resid}
    df_info = pd.concat(labeled, axis=1, names=["SplineInfo"])
    return df_info


def main_one_set_from_scratch(file_path):
    print("Started")
    # Generate the splines, save a DataFrame containing cubic splines objects
    df_splines, data, data_log, data_smooth = make_spline_dataframe(file_path)

    # Plot the splines against the data, showing spline knots.
    compare_splines_data(df_splines, data_log, data_smooth, showplots=False,
                do_knots=True, nice_name=nicer_name(file_path))

    df = knots_info(df_splines)
    print(df["NumberKnots"].mean(axis=1))
    print(df["Residuals"].mean(axis=1))


if __name__ == "__main__":
    # Choose the file to process
    filepath = os.path.join("data", "final", "cytokineConcentrationPickleFile"
        +"-20190608-PeptideComparison_OT1_Timeseries_19-final.pkl")
    main_one_set_from_scratch(filepath)
