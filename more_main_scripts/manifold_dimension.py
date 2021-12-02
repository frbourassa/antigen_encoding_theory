""" Small script to compute the Hausdorff dimension of the cytokine integrals
manifold. We hope to find something around 2. We use the correlation function
method and the Takens estimator to extract a good estimate of its slope for
small distances, while using the information available from larger distances.

To run this script, you need:
- Processed cytokine time series in the data/processed/ folder

It directly produces a figure used in the supplementary materials. 

@author:frbourassa
January 30, 2020
"""
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation

# Can execute from any folder and it still works with this path modification
import os, sys
main_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if main_dir_path not in sys.path:
    sys.path.insert(0, main_dir_path)

from ltspcyt.scripts.neural_network import import_WT_output


# This turns out to be very bad!
def compute_pairwise_distances_broadcast(x):
    """ Compute the matrix of pairwise euclidean distances between points.
    Will require D*N^2*8 bytes of memory at least, where x is NxD.
    This is in fact slower than the method with a for loop below,
    because of the broadcasting and copying of arrays, and because each
    distance is computed twice.

    Timing experiment:
        In [3]: pts = np.random.rand(1000*10).reshape(1000, 10)

        In [4]: %timeit compute_pairwise_distances(pts)
        21 ms ± 1.14 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

        In [6]: %timeit compute_pairwise_distances_broadcast(pts)
        61.1 ms ± 1.55 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    And it gets even worse  for larger matrices!
    (1.14s vs 4.5 s for 10 000 3d points) Surprising.

    Args:
        x (np.2darray); each row is a point; each column, a component.
    Returns:
        d_ij (np.2darray): element i, j is distance between x[i] and x[j].
    """
    # We compute them twice (ij and ji) vectorially because this is still
    # faster than a for loop on i or j.
    # Exploit broadcast, by making sure that columns (components) are always
    # in the last index; the new axes make sure each pair of points is taken.
    return np.sqrt(np.square(x[:, None, :] - x[None, :, :]).sum(axis=2))

def compute_pairwise_distances(x):
    """ Same as compute_pairwise_distances, but won't use more than
    max(2*D*N, N^2) memory, and turns out to be faster!
    Only the upper triangular part of the matrix is filled.
    """
    d_ij = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        d_ij[i, i+1:] = np.sqrt(np.square(x[i+1:] - x[i:i+1]).sum(axis=1))
    # Copy the other half? Don't bother
    return d_ij

def compute_correlation_cdf(pts, nr=None, r0=None):
    """ Compute the cumulative density function C(r) = average number of
    neighbors within distance r of a given point, at n r values, up to
    a distance r0 (determined automatically by default).

    Uses N^2*8 bytes of memory: this grows fast with N=pts.shape[0]!

    Args:
        pts (np.2darray): each row is a sample point; each column, a component.
        nr (int): number of r values at which to compute the cdf

    Returns:
        r_ax (np.1darray): distances at which the cdf C(r) was computed
        cr (np.1darray): the value of C(r) at each r.
    """
    # Compute all pairwise distances, store them in a flattened matrix.
    # Exclude the diagonal, we don't want zeros.
    pw_dist = compute_pairwise_distances(pts)
    pw_dist = pw_dist[np.triu_indices(pts.shape[0], k=1)]

    # Sort distances once, and keep track of index of distances already counted
    # Save time by not going through n times; better if log(N) < len(r_ax)
    pw_dist.sort()  # N^2*log(N^2)

    # Determine r0 if it isn't specified.
    # The sweet spot to avoid object size effects for something non-circular
    # seems to be r0 = 0.01*max pairwise distance (here sqrt(2)).
    if r0 is None:
        r0 = 0.01 * pw_dist[-1]
    if pw_dist[0] / pw_dist[-1] < 1e-16:
        print("May encounter ZeroDivide error")
    # Determine nr if it isn't specified. Aim for at least 50,
    # or one for each 5 sample pairs below r0.
    if nr is None:
        nr = max(np.sum(pw_dist <= r0) // 5, 50)
        print("Determined nr = ", nr)

    # Use a linear scale because we don't wan't to weight too much the
    # regime of very small distances, which will never have lots of samples.
    # Drop the smallest 0.1% of distances.
    r_ax = np.linspace(np.amin(pw_dist)*1.001, r0, nr)

    # For each r, compute the number of pairwise distances below that r.
    # Can do all this without a for loop: searchsorted for an array of values.
    # This gives the number of points <= r_ax[i] because of side="right"
    # with side="left" this is nb of points < r_ax[i].
    cr = np.searchsorted(pw_dist, r_ax, side="right")
    # Normalize by the number of pairs
    cr = cr / (pts.shape[0] * (pts.shape[0]-1))
    return r_ax, cr


def takens_estimate(cr, r_ax):
    """ Compute the Takens estimate of the dimension, based on C(r).

    Args:
        cr (np.1darray): correlation function C(r) at different points.
        r_ax (np.1darray): r values at which we have computed C(r).
    Returns:
        nu (float): Takens estimate of the dimension, which is a way to compute
            the average slope of lnC(r) vs ln(r) at small r.
        errnu (float): error on the Takens estimate, nu / sqrt(cr.size)
        b (float): intercept minimizing the linear log-log fit C(r) with
            the slope being nu.
    """
    dr = np.diff(r_ax)
    cr_over = cr / r_ax

    # Trapeze rule
    integral = 0.5 * np.sum(dr * (cr_over[1:] + cr_over[:-1]))
    nu = cr[-1] / integral

    # Corresponding intercept for plotting a linear fit.
    b = np.mean(np.log10(cr) - nu*np.log10(r_ax))

    # Error estimate
    errnu = nu / np.sqrt(cr.size)
    return nu, errnu, b

# Apply slope fitting and Takens estimates on some data
def dimension_correl(samples, nr=None, r0=None):
    """
    Args:
        samples (np.2darray): each row is a sample point;
            each column, a component.
        nr (int): number of r values at which to compute the cdf
        ro (float): cutoff distance.

    Returns:
        [dim_tak, err_tak, b_tak]: return of takens_estimate
        res (np.stats.LinregressResult): return object of linear regression
        [fig, ax]: figure and axis.
    """
    r_ax, c_r = compute_correlation_cdf(samples, nr=nr, r0=r0)

    # Fit a slope, should find a slope = 2 or close to it.
    logc_r = np.log10(c_r)
    logr_ax = np.log10(r_ax)
    res = sp.stats.linregress(logr_ax, logc_r)
    slp, intercept, reg_err = res.slope, res.intercept, res.stderr
    print("Slope fitting dimension estimate: {:.3f} \pm {:.3f}".format(slp, reg_err))

    # Compare to Takens estimate.
    dim_tak, err_tak, b_tak = takens_estimate(c_r, r_ax)
    print("Takens estimate: {:.3f} \pm {:.3f}".format(dim_tak, err_tak))
    # This seems to vary a lot less depending on trials, while fitting the
    # slope is very sensitive to the few points closest to min distance
    # (this varies a lot between trials because very few points are very close)

    # Plot the linear and Takens regressions
    fig, ax = plt.subplots()
    fig.set_size_inches(2.5, 2.5)

    ax.plot(logr_ax, logc_r, label="Data", color="grey")
    ax.plot(logr_ax, (slp * logr_ax + intercept), label="Lin. reg.",
            color="k", ls=":")
    ax.plot(logr_ax, (dim_tak * logr_ax + b_tak), label="Takens",
            color="k", ls="--")

    # ax.set(yscale="log")
    ax.annotate(r"$\hat{\nu}_{Tak} = $" + r"${:.1f} \pm {:.1f}$".format(dim_tak, err_tak),
                xy=(0.4, 0.2), xycoords="axes fraction", fontsize=8)
    ax.annotate(r"$\hat{\nu}_{reg} = $" + r"${:.1f} \pm {:.1f}$".format(slp, reg_err),
                xy=(0.4, 0.3), xycoords="axes fraction", fontsize=8)

    ax.set_xlabel(r"$\log_{10}(r)$ [-]", size=8)
    ax.set_ylabel(r"$\log_{10}C(r)$ [-]", size=8)
    leg = ax.legend(fontsize=8)
    leg.get_frame().set_linewidth(0.0)
    ax.tick_params(axis="both", labelsize=6, width=1., length=2.5)
    fig.tight_layout()

    return [dim_tak, err_tak, b_tak], res, [fig, ax]


if __name__ == "__main__":
    # Test on a square with well enough samples (10^4)
    # dimension_correl(np.random.rand(10000).reshape(5000, 2), nr=1000)

    # Import data
    cytokines = ["IFNg", "IL-2", "IL-17A", "IL-6", "TNFa"]
    df_data = pd.read_hdf(os.path.join(main_dir_path, "data", "processed",
                "PeptideComparison_OT1_Timeseries_20.hdf"))
    # Slice the desired values only
    df_data = df_data.loc["100k"]
    df_data = df_data.xs("integral", level="Feature", axis=1)
    df_data = df_data[cytokines]
    print(df_data.index.names)
    print(df_data.columns)

    # Exclude the first few hours, because this is probably just a small 5D bunch/ball.
    df_data = df_data.loc[df_data.index.isin(range(4, 75), level="Time")]
    print(df_data)

    # Determine the dimension of our cytokine manifold!
    [dim_tak, err_tak, b_tak], res, [fig, ax] = dimension_correl(df_data.values, r0=0.13)

    figname = os.path.join(main_dir_path, "figures", "supp",
                "hausdorff_correlation_PeptideComparison20.pdf")
    fig.savefig(figname, transparent=True)
    plt.show()
    plt.close()
