import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Add error bars; can't use relplot or catplot anymore,
# they don't have the option to include known standard deviations
def plot_params_vs_logec50(df_estim, df_estim_vari, ser_x, cols_plot=None,
                        ser_interp=None, x_name="Peptide", col_wrap=3, figax=None):
    """ Optional df_interp, which contains interpolating splines for each
    parameter as a function of the x variable (log EC50, usually).

    Code taken from mi_param_space/HighMI_3_channel_capacity.ipynb and improved
    for publication-quality figures, width 4.75 inches (Science).
    """
    if cols_plot is None:
        nplots = len(df_estim.columns)
        cols_plot = df_estim.columns
    else:
        nplots = len(cols_plot)

    nrows = nplots // col_wrap + min(1, nplots % col_wrap)  # Add 1 if there is a remainder.
    ncols = min(nplots, col_wrap)
    if figax is None:
        fig, axes = plt.subplots(nrows, ncols, sharey=False, sharex=True)
        fig.set_size_inches(4.75, 1.5 + 1.25*(nrows-1))
        axes = axes.flatten()
    else:
        fig, axes = figax
        axes = axes.flatten()

    for i in range(nplots):
        estim = df_estim[cols_plot[i]].copy()
        stds = np.sqrt(df_estim_vari[cols_plot[i]]).copy()
        x_labels = estim.index.get_level_values(x_name)
        xpoints = ser_x.reindex(x_labels)  # assume ser_x has a single index level?
        sort_how = np.argsort(xpoints)
        xpoints = xpoints[sort_how]
        x_labels = x_labels[sort_how]
        estim = estim.iloc[sort_how]
        stds = stds.iloc[sort_how]
        xmin, xmax = np.amin(xpoints), np.amax(xpoints)
        axes[i].set_xlim((xmin-0.12*(xmax-xmin), xmax+0.02*(xmax-xmin)))  # Room for last label
        axes[i].errorbar(xpoints, estim, yerr=stds, ls='none', marker="o",
            ms=2., label="Fit", mfc="k", mec="k", ecolor="k", elinewidth=1.)
        axes[i].invert_xaxis()
        axes[i].set_ylabel(cols_plot[i])
        ymin, ymax = np.amin(estim), np.amax(estim)
        # Make room for bottom and top labels
        ymin = ymin-0.12*(ymax-ymin)
        ymax = ymax + 0.12*(ymax-ymin)
        axes[i].set_ylim((ymin-0.02*(ymax-ymin), ymax+0.02*(ymax-ymin)))
        for j in range(len(x_labels)):
            ha = "left"
            if j == 0:  # N4
                xr = 0.02*(xmax - xmin)
                yr = 0.015*(ymax - ymin)
                va = "bottom"
            elif j == len(x_labels) - 1:  # E1
                xr = -0.02*(xmax - xmin)
                yr = -0.02*(ymax - ymin)
                va = "top"
            else:
                # If the spline to the right (smaller EC50) is decreasing to the next point,
                # put label below.
                if ser_interp is not None:
                    derivative = ser_interp[cols_plot[i]](xpoints[j-1]) - ser_interp[cols_plot[i]](xpoints[j])
                else:
                    derivative = estim.iloc[j-1] - estim.iloc[j]
                if derivative > 0:  # Label below
                    xr = 0.02*(xmax - xmin)
                    yr = -0.02*(ymax - ymin)
                    va = "top"
                # If the line is decreasing, label above
                else:
                    xr = -0.02*(xmax - xmin)
                    yr = 0.02*(ymax - ymin)
                    va = "bottom"
            axes[i].annotate(x_labels[j], xy=(xpoints[j], estim[j]+yr), fontsize=6, va=va, ha=ha, color="grey")
        if ser_interp is not None:
            spl = ser_interp[cols_plot[i]]
            xrange = np.linspace(xmin, xmax, 201)
            axes[i].plot(xrange, spl(xrange), lw=1.5, label="Interpol.", color="xkcd:royal blue")
            if i == 0:
                axes[i].legend(handlelength=1.5, handletextpad=0.5, borderpad=0.5, fontsize=6)
    return [fig, axes]
