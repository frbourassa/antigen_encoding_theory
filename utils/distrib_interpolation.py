""" Module containing functions previously defined in notebooks,
which are used to interpolated multivariate normal parameters as a function
of EC50.

@author: frbourassa
Fall 2020
"""
import numpy as np
import pandas as pd
import scipy as sp
from scipy import interpolate

from utils.statistics import estimate_empirical_covariance, build_symmetric, cholesky_variance


def stats_per_levels(df_p, levels_groupby, feats_keep):
    """ Compute a mean vector and a covariance matrix for the features (columns)
    specified in feats_keep, with a different matrix for each tuple entry in the
    levels_groupby, keeping points of different labels in other index levels as
    different samples.

    Args:
        df_p (pd.DataFrame): columns are features (model parameters for our application)
            while index labels the different conditions/subpopulations/independent variable.
            df_p.columns should be an Index, not a MultiIndex.
        levels_groupby (list): list of index level names by which to group data,
            each entry in those levels considered a different population with its own covariance.
        feats_keep (list): list of features/columns to use as random variables;
            other columns are not considered.

    Returns:
        df_means (pd.DataFrame): DataFrame with index levels being those in levels_groupby,
            and columns being the average of the variables in feats_keep.
        df_means_estim_vari (pd.DataFrame): variance on each estimator in df_means
            (just the variance of each variable divided by the number of points).
        df_covs (pd.DataFrame): DataFrame with index levels being those in levels_groupby,
            and columns being upper triangular part of the covariance matrix entries,
            flattened, with variables in the order specified in feats_keep.
        df_covs_estim_vari (pd.DataFrame): variance on each estimator in df_covs
        ser_nelem (pd.Series): the number of sample points used to estimate each distrib.

    Example: feats_keep = ["x1", "x2"]; covariance matrix would be
        [x1^2  x1*x2
         x2*x1  x2^2]
    so the three columns in the returned df_covs would be [x1^2, x1*x2, x2^2]
    """
    ## Check that all levels_groupby exist and all features in feats_keep exist too.
    for lvl in levels_groupby:
        if lvl not in df_p.index.names:
            raise ValueError("Invalid index level:", lvl)
    if not isinstance(df_p.columns, pd.Index):
        raise TypeError("Make sure df_p is sliced until columns are an Index")
    for fet in feats_keep:
        if fet not in df_p.columns.values:
            raise ValueError("Invalid column:", lvl)

    ## Pre-build the final DataFrames
    # Columns: pairs of variables
    nft = len(feats_keep)
    cov_entries = []
    for i in range(nft):
        cov_entries += [feats_keep[i]+"*"+feats_keep[j] for j in range(i, nft)]

    # Rows: drop all levels not to be kept, then take .unique() to have once each entry
    levels_remove = list(set(df_p.index.names).difference(levels_groupby))
    cov_index = df_p.index.droplevel(levels_remove).unique()

    # Prepare DataFrames of covariance matrices
    df_covs = pd.DataFrame(np.zeros([len(cov_index), nft*(nft+1) // 2]),
                           index=cov_index, columns=cov_entries)
    df_covs_estim_vari = df_covs.copy()  # To hold the variance of each estimator
    df_covs.columns.name = "Covariance element"
    df_covs_estim_vari.columns.name = "Var[Cov estimator]"

    # Prepare DataFrames of means
    df_means = pd.DataFrame(np.zeros([len(cov_index), nft]),
                           index=cov_index, columns=feats_keep)
    df_means_estim_vari = df_means.copy()
    df_means.columns.name = "Mean element"
    df_means_estim_vari.columns.name = "Var[Mean estimator]"

    # Prepare the Series of number of sample points
    ser_nelem = pd.Series(index=cov_index, dtype=int)
    ser_nelem.name = "N sample points"

    ## Loop over entries in levels_to_keep and compute the
    # mean and covariance matrix of variables in feats_keep
    for gp in df_p[feats_keep].groupby(levels_groupby):
        # A groubpy object is a tuple with key, values, ...
        mat_cov, mat_varicov = estimate_empirical_covariance(gp[1].values)
        tri_ind = np.triu_indices(nft)
        df_covs.loc[gp[0]] = mat_cov[tri_ind].flatten()
        df_covs_estim_vari.loc[gp[0]] = mat_varicov[tri_ind].flatten()

        # Sample mean and its variance
        df_means.loc[gp[0]] = gp[1].mean()
        # Count the number of non-NA elements in each feature
        n_elem = gp[1].count(axis=0).min()
        print("{}:{:4d} elements".format(gp[0], n_elem))
        df_means_estim_vari.loc[gp[0]] = mat_cov[np.diag_indices(nft)] / n_elem
        ser_nelem[gp[0]] = n_elem

    return df_means, df_means_estim_vari, df_covs, df_covs_estim_vari, ser_nelem


def compute_cholesky_dataframe(df_covs, ser_npts):
    """ Compute, for each covariance matrix in df_covs, the
    Cholesky decomposition estimator $\hat{L}$ and its variance.
    Return them in separate dataframes with the same index
    as df_covs. The formulas for the variance are given
    in Olkin, 1985, Estimating a Cholesky Decomposition.
    We use a biased version of $\hat{L}$ which directly gives
    $\hat{L} \hat{L}^T = \hat{\Sigma}$. The variance formulas
    developed by Olkin only depend on L itself, and are approximated
    with our biased $\hat{L}$, which makes them biased as well,
    but convergent nevertheless (I think).
    Anyways, it will be good enough for my purpose of
    reconstructing $\Sigma$ after interpolation.


    Args:
        df_covs (pd.DataFrame): each row is a different condition,
            each column is a pair of RVs (an element of \Sigma),
            ordered as the upper triangular part of \Sigma flattened
            in C-order (row-first). So this is (for d dimensions):
            "X_1*X_1", "X_1*X_2", "X_1*X_3", ..., "X_d*X_d".
            Columns should be and Index, not  MultiIndex, as they will
            be sorted in-place in this order.
        ser_npts (pd.Series): the number of population samples used
            to estimate the covariance matrix in each row of df_covs.

    Returns:
        df_chol (pd.DataFrame): Cholesky decomposition of each
            covariance matrix in df_covs. Columns are ordered
            as the lower triangular elements of L (the other L
            elements are zero) flattened in C-order (row_first).
            So this is L_{11}, L_{21}, L_{22}, ..., L_{dd}
            This is the order returned by np.tril_indices.

            Same index as df_covs.
        df_chol_vari (pd.DataFrame): the variance of each estimated
            Cholesky matrix element. Same indexing as df_chol.
    """
    # Extract the dimension
    dim = int(-0.5 + 0.5*np.sqrt(1 + 8*len(df_covs.columns)))

    # Make sure the columns are ordered as we expect. See for the 1-liner:
    # https://stackoverflow.com/questions/30083947/how-to-convert-list-of-lists-
    # to-a-set-in-python-so-i-can-compare-to-other-sets
    concat = [a for lbl in df_covs.columns
                  for a in lbl.split("*")]
    params = set(concat)
    # Sort parameters in order of appearance
    params = sorted(list(params), key=concat.index)
    # Build expected index:
    # Outer loop is the first one, so 1st param (a) varies slowest
    expected_cols = [params[i]+"*"+b for i in range(len(params))
                                         for b in params[i:]]
    # Sort the columns as expected, so we can use .values confidently
    df_covs = df_covs.reindex(expected_cols, axis=1)

    # Initialize the container dataframes
    lower_triang_params = [params[i]+"*"+b for i in range(len(params))
                                               for b in params[:i+1]]
    chol_columns = pd.Index(lower_triang_params, name="Cholesky element")
    df_chol = pd.DataFrame(index=df_covs.index.copy(), columns=chol_columns, dtype=np.float64)
    df_chol_vari = pd.DataFrame(index=df_covs.index.copy(), columns=chol_columns.copy(), dtype=np.float64)
    df_chol_vari.columns.name = "Var[Chol estimator]"

    # Treat one matrix at a time
    lower_ind = np.tril_indices(dim)
    for k in df_covs.index:
        # Build the full covariance matrix from the upper triangular part
        # and take the Cholesky decomposition
        try:
            chol = np.linalg.cholesky(build_symmetric(df_covs.loc[k].values))
        except np.linalg.LinAlgError as e:
            # There is probably a slightly negative eigenvalue
            print("Could not decompose index {}".format(k))
            print(e)
            print("Trying to add +1e-8 to the diagonal")
            try:
                chol = np.linalg.cholesky(build_symmetric(df_covs.loc[k].values) + np.identity(dim)*1e-8)
            except np.linalg.LinAlgError:
                print("It failed; skipping this value")
                continue
            else:
                print("It worked!")

        # Store with tril_indices
        df_chol.loc[k] = chol[lower_ind]

        # Compute the variance estimators
        chol_vari = cholesky_variance(chol, ser_npts[k])
        df_chol_vari.loc[k] = chol_vari[lower_ind]

    return df_chol, df_chol_vari



def interpolate_params_vs_logec50(df_estim, df_estim_vari, ser_x, x_name="Peptide"):
    """ Given a DataFrame of parameter values (one per column) for different values
    of the input variable (whose name is x_name and values are in ser_x),
    find an interpolation, for each condition in the rows of df_estim
    """
    df_splines = pd.Series(index=df_estim.columns, dtype=object)
    for k in df_estim.columns:
        pvals = df_estim.xs(k, axis=1)
        pvari = df_estim_vari.xs(k, axis=1)
        x_labels = [x for x in list(pvals.index.get_level_values(x_name).unique()) if x in ser_x.index]
        x_labels = sorted(list(x_labels), key=lambda y:ser_x[y])  # Sort by increasing x value

        # Make sure the values match the order of the parameter values
        pvals = pvals.reindex(x_labels)
        pvari = pvari.reindex(x_labels)
        xpoints = ser_x.reindex(x_labels)

        # Weight each point with 1/sqrt(variance), then the smoothing factor s can be left to default.
        spl = sp.interpolate.UnivariateSpline(xpoints.values, pvals.values, w=1.0/np.sqrt(pvari.values), s=0.2*len(pvari.values))

        # Now, make a PCHIP interpolator, monotonous, on the value of this smoothed spline
        # evaluated at the empirical x values.
        # Avoids having nasty, non-monotonous artifacts between the empirical points.
        yvals = spl(xpoints.values)
        spl = sp.interpolate.PchipInterpolator(xpoints, yvals)
        df_splines[k] = spl

    return df_splines


def eval_interpolated_means_covs(ser_interp_m, ser_interp_covs, ser_interp_chol,
                                 xvals, nins, ndim, epsil=1e-5):
    """ Returns an array of mean vectors and covariance matrices computed from the
    interpolations in ser_interp_m and ser_interp_covs, evaluated at each input value
    in xvals.

    Args:
        ser_interp_m (pd.Series): Series containing a Spline object per parameter (per dimension)
        ser_interp_covs (pd.Series): Series containing a Spline per pair of parameters
        ser_interp_chols (pd.Series): Series containing a Spline per pair of parameters
            in the Cholesky decomposition of the matrix interpolated in ser_interp_covs.
        xvals (np.1darray): input variable values at which to evaluate the splines.
        nins (int): number of inputs = number of values in xvals
        ndim (int): number of dimensions = number of parameters.
        epsil (float): minimal positive value for diagonal elements of Cholesky decomposition.
    """
    # Checks:
    if len(ser_interp_m.index) != ndim:
        raise ValueError("There should be one entry in ser_interp_m per dimension")
    elif len(ser_interp_covs) != ndim * (ndim + 1) / 2:
        raise ValueError("There should be dim*(dim+1)/2 (number of pairs) entries in ser_interp_covs")
    elif len(ser_interp_chol) != ndim * (ndim + 1) / 2:
        raise ValueError("There should be dim*(dim+1)/2 (number of pairs) entries in ser_interp_chol")
    elif len(xvals) != nins:
        raise ValueError("there should be nins input values in xvals")
    elif epsil < 0.:
        raise ValueError("epsil should be a small (<<1) positive float")

    # Initialize containers.
    covs = np.zeros([nins, ndim, ndim])
    covs_comparison = covs.copy()
    means = np.zeros([nins, ndim])

    # Fill the matrices with the interpolated values
    param_labels = list(ser_interp_m.index)  # Always use this order
    upper_indices = np.triu_indices(ndim)
    for j in range(nins):
        x_j = xvals[j]
        # Initialize this mean and covariance
        meanvec = np.zeros(ndim)
        cov_flat_triang = np.zeros(int(ndim * (ndim + 1))//2)
        cholT_flat_triang = cov_flat_triang.copy()

        # Loop over parameters
        # Index for next pair of parameters = n(i-1) + (i-1)i/2 + (k-i) if i >= 1. Easier to increment!
        idx_pair = 0
        for i in range(ndim):
            pi = param_labels[i]
            meanvec[i] = ser_interp_m[pi](x_j)

            # Loop over remaining pairs of parameters for Cholesky and direct cov
            # since the Cholesky is lower triangular, index as pk*pi, where
            # the upper triangular part of covariance is pi*pk
            for k in range(i, ndim):
                pk = param_labels[k]
                spl_cov = ser_interp_covs[str(pi)+"*"+str(pk)]
                spl_chol = ser_interp_chol[str(pk)+"*"+str(pi)]
                cov_flat_triang[idx_pair] = spl_cov(x_j)
                # Enforce positive definiteness by clipping diagonal elements
                # to some small positive value epsil.
                if k == i:
                    cholT_flat_triang[idx_pair] = max(epsil, spl_chol(x_j))
                else:
                    cholT_flat_triang[idx_pair] = spl_chol(x_j)
                idx_pair += 1

        # Then, put the current mean and direct covariance matrix in the container
        means[j] = np.copy(meanvec)
        covs_comparison[j] = np.asarray(build_symmetric(cov_flat_triang))

        # Also build the proper covariance matrix from the Cholesky^T
        cholT = np.zeros([ndim, ndim])
        cholT[upper_indices] = cholT_flat_triang
        covs[j] = np.dot(cholT.T, cholT)

    return means, covs, covs_comparison
