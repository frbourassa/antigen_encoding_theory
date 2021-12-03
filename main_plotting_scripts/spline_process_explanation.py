# -*- coding:utf-8 -*-
""" Short script to generate a few figures detailing the two steps of
the data smoothing used for the cytokine integral classifier.

To run this script, you need raw cytokine time series in the data/final/ folder.
You can change which dataset is plotted in the __main__ block at the end of the script.

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
if main_dir_path not in sys.path:
    sys.path.insert(0, main_dir_path)

# Slightly different process_file (legacy version) that returns spline objects
# and dataframes shaped differently from usual version
from utils.plotting_splines import nicer_name, process_file

plt.rcParams["figure.figsize"] = [6.5, 6]
plt.rcParams["lines.linewidth"] = 2.
plt.rcParams["font.size"] = 12.
plt.rcParams["axes.labelsize"] = 12.
plt.rcParams["legend.fontsize"] = 12.
plt.rcParams["xtick.labelsize"] = 12.
plt.rcParams["ytick.labelsize"] = 12.


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
    ax.set_xlabel("Time (h)")
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
    ax.set_xlabel("Time (h)")
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
    pklfile = "cytokineConcentrationPickleFile-20190608-PeptideComparison_3-final.hdf"
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
