""" Streamlined version of my local script for computing EMD on PC2 axis.
@author: frbourassa
April 2021
"""
import numpy as np
import scipy as sp
import pandas as pd

from scipy.stats import wasserstein_distance, ks_2samp
from utils.statistics import principal_component_analysis
from metrics.earth_mover import compute_emd
from metrics.kldiv import KLdivergence

# Function to compute the chosen distance, formatting data accordingly
def appropriate_dist(d1, d2, method):
    """ Input 2D arrays, flattened for the methods which require it. """
    if method == "EMD" and d1.shape[1] == 1:
        dst = wasserstein_distance(d1.ravel(), d2.ravel())
    elif method == "EMD":
        dst = compute_emd(d1, d2)
    elif method == "KL_div":
        dst = KLdivergence(d1, d2)
    elif method == "K-S" and d1.shape[1] == 1:
        dst = ks_2samp(d1.ravel(), d2.ravel())[0]  # drop the p-value
    elif method == "K-S":
        raise ValueError("Cannot apply Kolmogorov-Smirnov to multivariate")
    else:
        raise NotImplementedError("We do not know distance " + method)
    return dst

def compute_distance_panel_PCA(df, pca_axes, dist_fct="EMD", ref_lbl="Null",
    lvl="Drug", intra_lvl="Peptide", intra_pair=None):
    """ Returns an array of distance values and a list of labels.
    Assumes the projection gives a 1D distribution, for simplicity.
    Known distance methods dist_fct:
        "EMD": earth mover's distance
        "KL_div": Kullback-Leibler divergence
        "K-S": Kolmogorov-Smirnov
     """
    drugs = list(df.index.get_level_values(lvl).unique())
    data_ref = df.xs(ref_lbl, level=lvl).values.dot(pca_axes)
    drugs.remove(ref_lbl)
    drugs.insert(0, ref_lbl)

    # Compare the ref. label against all others. The self-distance is zero.
    dist_panel = np.zeros(len(drugs) + 1)
    for i in range(1, len(drugs)):
        data_i = df.xs(drugs[i], level=lvl).values.dot(pca_axes)
        dist_panel[i] = appropriate_dist(data_ref, data_i, dist_fct)

    # As a reference, add the median distance between intra_level in
    # the ref. label. Unless an intra_pair has been forced.
    data_ref = df.xs(ref_lbl, level=lvl)
    if intra_pair is None:
        intra_peps = data_ref.index.get_level_values(intra_lvl).unique()
        intra_dsts, intra_labels = [], []
        for i, ipep in enumerate(intra_peps):
            data_i = data_ref.xs(ipep, level=intra_lvl).values.dot(pca_axes)
            for j in range(i+1, len(intra_peps)):
                jpep = intra_peps[j]
                data_j = data_ref.xs(jpep, level=intra_lvl).values.dot(pca_axes)
                intra_dsts.append(appropriate_dist(data_i, data_j, dist_fct))
                intra_labels.append(ipep + " vs " + jpep)
        # Append the median of those to drugs and dist_panel
        sort_index = np.argsort(intra_dsts)
        intra_dsts = np.asarray(intra_dsts)[sort_index]
        intra_labels = [intra_labels[i] for i in sort_index]
        dist_panel[-1] = intra_dsts[len(intra_dsts) // 2]
        drugs.append(intra_labels[len(intra_dsts) // 2])
    else:
        dist_panel[-1] = appropriate_dist(
            data_ref.xs(intra_pair[0], level=intra_lvl).values.dot(pca_axes),
            data_ref.xs(intra_pair[1], level=intra_lvl).values.dot(pca_axes),
            dist_fct
        )
        drugs.append(" vs ".join(intra_pair))
    return dist_panel, drugs
