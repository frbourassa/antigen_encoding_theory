# -*- coding:utf-8 -*-
""" Short script to generate a few figures detailing the two steps of
the data smoothing used for the cytokine integral classifier,
at least up to August 2019.
@author:frbourassa
September 1, 2019
"""
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns
import pandas as pd
sns.reset_orig()

import os, sys
main_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, main_dir_path)

from ltspcyt.scripts.process_raw_data import (treat_missing_data,
    lod_import, log_management, smoothing_data, generate_splines)

plt.rcParams["figure.figsize"] = [6.5, 6]
plt.rcParams["lines.linewidth"] = 2.
plt.rcParams["font.size"] = 12.
plt.rcParams["axes.labelsize"] = 12.
plt.rcParams["legend.fontsize"] = 12.
plt.rcParams["xtick.labelsize"] = 12.
plt.rcParams["ytick.labelsize"] = 12.

### Small utility functions
def nicer_name(fname):
    """ Extract a nice name for the Series from the raw file path.

    Args:
        fname (str): the file path
    Returns:
        (str): a nicer, shorter name.
    """
    folderlst = os.path.split(fname)
    fname = folderlst[-1]
    fragments = fname.split(sep="-")
    # Search for the date; the fragment after is the good one.
    idx = 0
    for frag in fragments:
        idx += 1
        if frag.isnumeric():
            break  # stop looping; the next one is the good one
    try:
        good_name = fragments[idx]
    except IndexError:
        print("No date was found in the file name; using last part after -")
        good_name = fragments[-1]
    # Make sure that this fragment ends with .pkl;
    # otherwise we made a false split, there was a - in the desired name
    # So add all the remaining parts until we hit the pkl name or the end
    else:
        while not good_name.endswith(".pkl") and idx < len(fragments) - 1:
            if fragments[idx + 1] not in ["modified.pkl", "final.pkl"]:
                good_name += fragments[idx + 1]
            else:
                good_name += ".pkl"
            idx += 1

    # In case there is still no .pkl in the name (other file type)
    try:
        where_to_cut = good_name.index(".")
    except:
        pass
    else:
        good_name = good_name[:where_to_cut]

    return good_name

### Processing function copied and modified here to return df of spline objects
def process_file(folder,file, **kwargs):
    """ Function to process the raw cytokine concentrations time series:
    Find missing data points and linearly interpolate between them, take log, rescale and smooth with a moving average, interpolate with cubic splines, and extract features (integral, concentration & derivatives) at desired times
    Also tries to load limits of detection

    Args:
        data_file (str): path to the raw data file (a pickled pd.DataFrame)

    Keyword args:
        take_log (bool): True to take the log of the concentrations in the
            preprocessing, False if the networks have to deal with raw values.
            Default: True.
        rescale_max (bool): True: rescale concentrations by their maximum to
            account for experimental variability, False if we postpone
            normalization to a later stage.
            Default: False.
        smooth_size (int, default=3): number of points to consider when
            performing the moving average smoothing on the data points.
            In other words, width of the kernel.
        rtol_splines (float): tolerance for spline fitting: specify the
            fraction of the sum of squared residuals between the raw data
            and the data smoothed with a moving average that will be used
            as the total error tolerance in UnivariateSpline. Default: 1/2
        max_time (float): last time point to sample from splines.

    Returns:
        data (pd.DataFrame): the rearranged raw data, before processing.
        data_log (pd.DataFrame): the normalized log time series
        data_smooth (pd.DataFrame): log data after applying a moving average
        spline_frame (pd.DataFrame): spline objects
        df (pd.DataFrame): concentrations, integrals, etc.
    """
    # Processing-related keyword arguments
    take_log = kwargs.get("take_log", True)
    rescale_max = kwargs.get("rescale_max", False)
    smooth_size = kwargs.get("smooth_size", 3)
    rtol_splines = kwargs.get("rtol_splines", 1/2)
    max_time = kwargs.get("max_time", 72)

    # Import raw data
    data = pd.read_pickle(os.path.join(folder, file))

    # Put all timepoints for a given cytokine in continuous columns
    data = data.stack().unstack('Cytokine')

    # Check for randomly or structurally missing datapoints and interpolate between them
    data = treat_missing_data(data)

    # Import the limits of detection, if any
    cytokine_lower_lod = lod_import(file[32:40])

    # Take the log of the data if take_log, else normalize in linear scale
    data_log = log_management(data, take=take_log, rescale=rescale_max, lod=cytokine_lower_lod)

    # Smooth the data points before fitting splines for interpolation
    data_smooth = smoothing_data(data_log, kernelsize=smooth_size)

    # Fit cubic splines on the smoothed series
    spline_frame = generate_splines(data_log, data_smooth,rtol=rtol_splines)

    # Return data in various stages of processing
    return [data, data_log, data_smooth, spline_frame]


### Main plotting function
def plot_spline_process_steps(chosen, df_raw, df_log, df_smooth, df_spline):
    """ Produces a panel of 2x2 plots, to be read from left to right first.
    1) The first column shows the raw data with a logarithmic axis.
    2) The second shows the raw data after taking
    the log and rescaling the min with the LOD and the max with the maximum
    concentration across all conditions in that dataset.
    versus the smoothing average. The second column shows the splines versus
    the smoothed data. It also shows the spline knots in red.

    Args:
        chosen (list): Condition chosen. Contains first the cytokine (str),
            then a tuple specifying the other index levels to select one
            specific condition, usually (Peptide, Concentration).
        df_raw (pd.DataFrame): the data as sent by Sooraj, with the indexes
            rearranged.
        df_log (pd.DataFrame): the df with the log + rescaled data
        df_smooth (pd.DataFrame): the log data after moving average smoothing
        df_spline (pd.DataFrame): df containing all splines for the dataset
    Returns:
        fig (matplotlib.figure.Figure): the figure
        axes (np.2darray): the four axes, in a 2x2 grid.
    """
    # Extract a few useful variables
    cond, cytokine = chosen
    fig, axes = plt.subplots(2, 2, sharex=True)
    # fig.set_size_inches(2*8, 2*6)
    spline = df_spline.loc[cond, cytokine]

    # Time axes. Add (0, 0) to the rescaled data
    exp_times = np.array(df_raw.columns.get_level_values("Time").unique())
    exp_times0 = np.concatenate(([0], exp_times))
    spline_times = np.linspace(exp_times0[0], exp_times0[-1], 201)
    spline_knots = spline.get_knots()

    # y values for each curve
    yraw = df_raw.loc[cond, cytokine]
    ylog = np.concatenate(([0], df_log.loc[cond, cytokine]))
    ysmooth = np.concatenate(([0], df_smooth.loc[cond, cytokine]))
    yspline = spline(spline_times)
    yknots = spline(spline_knots)

    # Curve styles
    style_raw = dict(ls="--", color="k", marker="o", ms=5, lw=2.)
    style_spline = dict(color="xkcd:royal blue", ls="-", lw=3.)
    style_knots = dict(ls="none", marker="^", ms=5, mfc="r", mec="k", mew=1.)
    style_smooth = dict(color="grey", ls=":", marker="s", ms=5, lw=2.)

    ## Plot raw data
    ax = axes[0, 0]
    ax.plot(exp_times, yraw, **style_raw, label="Raw")
    #ax.set_yscale("log")
    ax.set_ylabel(cytokine + " [nM]")
    ax.set_title("A", loc="left")
    ax.legend()

    ## Plot rescaled + smoothing + spline
    ax = axes[1, 1]
    ax.plot(exp_times0, ylog, **style_raw, label="Log")
    ax.plot(exp_times0, ysmooth, **style_smooth, label="Smoothed")
    ax.plot(spline_times, yspline, **style_spline, label="Cubic spline")
    ax.plot(spline_knots, yknots, **style_knots, label="Spline knots")
    ax.set_xlabel("Time [h]")
    ylbl = r"$\log_{10}($" + cytokine + r"$ / \mathrm{LOD})$"
    ax.set_ylabel(ylbl)
    ax.legend()
    ax.set_title("D", loc="left")
    # Use the same limits for the remaining two plots
    ylims = ax.get_ylim()

    ## Rescaled data
    ax = axes[0, 1]
    ax.plot(exp_times0, ylog, **style_raw, label="Log")
    ax.set_title("B", loc="left")
    # Same axes for all three plots after rescaling
    ax.set_ylim(ylims)
    ax.legend()
    ax.set_ylabel(ylbl)
    ## Rescaled data + smoothing average
    ax = axes[1, 0]
    ax.plot(exp_times0, ylog, **style_raw, label="Log")
    ax.plot(exp_times0, ysmooth, **style_smooth, label="Smoothed")
    ax.set_ylabel(ylbl)
    ax.set_xlabel("Time [h]")
    # Same axes for all three plots after rescaling
    ax.set_ylim(ylims)
    ax.set_title("C", loc="left")
    ax.legend()

    fig.tight_layout()
    return fig, axes

if __name__ == "__main__":
    # Processing arguments
    process_args = dict(
        take_log = True,
        rescale_max = False,
        smooth_size = 3,
        rtol_splines = 1/3,
    )

    # Choose the data file and import it.
    folder = os.path.join(main_dir_path, "data", "final")
    #pklfile = "cytokineConcentrationPickleFile-20190412-PeptideComparison_OT1_Timeseries_18-final.pkl"
    #pklfile = "cytokineConcentrationPickleFile-20190718-PeptideComparison_OT1_Timeseries_20-final.pkl"
    pklfile = "cytokineConcentrationPickleFile-20190608-PeptideComparison_OT1_Timeseries_19-final.pkl"
    ret = process_file(folder, pklfile, **process_args)
    [data, data_log, data_smooth, spline_frame] = ret

    # Choose a condition and a cytokine.
    print(spline_frame.index.names)
    chosen_conditions = [("100k", "N4", "10nM"), "IL-2"]
    print("Minimum value (LOD)", data.loc[:, chosen_conditions[1]].min())
    # chosen_conditions = [("100k", "Q4", "10nM"), "TNFa"]

    # See the following function for the details of the plot
    fig, axes =  plot_spline_process_steps(chosen_conditions, data, data_log,
        data_smooth, spline_frame)
    nicename = nicer_name(pklfile)
    figname = os.path.join(main_dir_path, "figures", "supp",
        "processing_steps_{0[1]}_{0[2]}_{1}_{2}.pdf".format(
        *chosen_conditions, nicename))
    fig.savefig(figname, format="pdf", transparent=True, bbox_inches="tight")
    plt.close()
