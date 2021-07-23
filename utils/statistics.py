""" Module with a few home-made statistics functions.

# Basic covariance estimator

## Diagonal terms: variance estimator

From RHB, we have the correct results:

$$ \hat{\sigma^2} = \frac{N}{N-1} s^2 = \frac{1}{N-1} \sum_i (x_i - \bar{x})^2
    = \frac{N}{N-1} \left(\overline{x^2} - \bar{x}^2 \right) $$

$$ \mathrm{Var}[\hat{\sigma^2}] = \frac{1}{N} \left(n_4 - \frac{N-3}{N-1} \hat{\sigma^2}^2 \right) $$

where $n_4$ is the fourth sample moment,

$$ n_4 = \frac1N \sum_i (x_i - \bar{x})^4 = \overline{x^4} - 4 \overline{x^3} \bar{x}
    + 6 \left(\overline{x^2} \right)^2 - 3 \bar{x}^4$$

## Off-diagonal terms: covariance
Use the empirical, unbiased estimator. That's the maximum likelihood estimator
times a factor $N/(N-1)$: $V_{\mathrm{emp}} = V_{ML} \frac{N}{N-1}$.

The empirical estimator is, explicitly,

$$ \hat{\mathrm{Cov}} [X, Y] = V_{xy} = \frac{N}{N-1} (\bar{xy} - \bar{x}\bar{y})
    = \frac{1}{N-1} \left( \sum_i x_i y_i - \frac{1}{N} \sum_i x_i \sum_j y_j \right) $$

Note that I defined $V_{xy}$ here with the right factor built in to remove bias
(often $V_{xy}$ is defined as just $\bar{xy} - \bar{x}\bar{y}$).

The variance of this estimator is quite complicated, but I obtained an expression
(which seems right, see below: it matches bootstrapping even for low $N$)
that at least has the right behaviour of decaying as $1/N$.
When we don't know the true moments of the distribution,
we can estimate it from the sample moments
$\bar{x}$, $\bar{y}$, $\bar{xy}$, $\bar{x^2}$, $\bar{y^2}$, $\bar{x^2y}$,
$\bar{xy^2}$, $\bar{x^2 y^2}$:

$$ \mathrm{Var}[V_{xy}] = \frac1N \overline{x^2 y^2} - \frac{N-2}{N(N-1)} {\overline{xy}}^2
+ \frac{1}{N(N-1)} \bar{x^2}\bar{y^2} + \frac{2(3N-4)}{N(N-1)} \overline{xy}\bar{x}\bar{y}
- \frac{2(2N-3)}{N(N-1)} \bar{x}^2 \bar{y}^2
- \frac{2(N-2)}{N(N-1)} \left( \overline{x^2 y}\bar{y} + \overline{x y^2} \bar{x} \right)
+ \frac{N-4}{N(N-1)} \left(\bar{x^2}\bar{y}^2 + \bar{x}^2 \bar{y^2} \right)$$


@author: frbourassa
Date: September 25, 2020
"""
import numpy as np
from scipy.special import gammaln  # for the Cholesky estimator variance

def estimate_empirical_covariance(samp, do_variance=True):
    """ Use the unbiased empirical sample covariance as the estimator of the true covariance
    between each pair of variables (columns) in samp, and build the covariance matrix from them.

    Args:
        samp (np.array): nxp matrix for n samples of p dimensions each.
            Pass the values of a dataframe for proper slicing.
        do_variance (bool): if False, do no compute the variance of the estimator.
            An array of zeros is returned instead (for consistency of returns).
            Useful for computing bootstrap replicates, for instance.
            Default: True.

    Returns:
        cov (np.array): the estimated covariance matrix, shape pxp.
        variances (np.array): the variance on each entry of cov, shape pxp.
    """
    p = samp.shape[1]  # Number of variables, p
    N = samp.shape[0]  # Number of points
    cov = np.zeros([p, p])

    # Compute useful moments and central moments of each variable
    # Avoids computing them many times,
    # and low memory requirement O(p) (versus O(np) for samp)
    m1 = np.mean(samp, axis=0)  # first moment, \bar{x}
    m2 = np.mean(samp**2, axis=0)  # \bar{x^2}

    # First, compute the diagonal terms: usual variances
    # Can extract all diagonal terms to 1D array and use element-wise
    cov[np.diag_indices(p)] = N / (N - 1) * (m2 - m1**2)

    # Second, compute the estimate of covariances (upper triangular part of the symmetric matrix)
    for i in range(p):
        x = samp[:, i]
        for j in range(i+1, p):
            y = samp[:, j]
            vxy = N / (N - 1) * (np.mean(x*y) - m1[i]*m1[j])
            cov[i, j] = vxy
            cov[j, i] = vxy  # fill the other half of the symmetric matrix

    # Third, compute the variance of the diagonal terms
    variances = np.zeros([p, p])
    if not do_variance:
        return cov, variances

    # Else: no need for the indent because we returned already
    # 1/N(n_4 - (N-3)/(N-1) (sigma^2)^2)
    n4 = np.mean((samp - m1[None, :])**4, axis=0)  # \frac1N \sum_i (x_i - \bar{x})^4
    variances[np.diag_indices(p)] = (n4 - (N-3)/(N-1) * cov[np.diag_indices(p)]**2) / N

    # Fourth, compute the variance of off-diagonal terms.
    for i in range(p):
        x = samp[:, i]
        for j in range(i+1, p):
            y = samp[:, j]
            x2y2 = np.mean(x**2 * y**2)
            x2y = np.mean(x**2 * y)
            xy2 = np.mean(x * y**2)
            xy = np.mean(x * y)
            nn1 = N * (N - 1)
            varvxy = x2y2 / N - (N-2)/nn1 * xy**2 + m2[i]*m2[j]/nn1
            varvxy += 2*(3*N - 4)/nn1 * xy * m1[i] * m1[j] - 2 * (2*N - 3)/nn1 * m1[i]**2 * m1[j]**2
            varvxy -= 2*(N-2)/nn1 * (x2y * m1[j] + xy2 * m1[i])
            varvxy += (N-4)/nn1 * (m2[i] * m1[j]**2 + m1[i]**2 * m2[j])
            variances[i, j] = varvxy
            variances[j, i] = varvxy

    return cov, variances


def bootstrap_empirical_covariance(samp, f=1., seed=None, nboot=1000):
    """ Use bootstrapping to estimate the covariance matrix, and the variance on that estimator.
    Will call repeatedly estimate_empirical_covariance with do_variance=False (the variance
    will be computed from the bootstrap replicates).

    Args:
        samp (np.ndarray): nxp matrix for n samples of p dimensions each.
            Pass the values of a dataframe for proper slicing.
        f (float): fraction of the n samples to use per bootstrap replicate.
        nboot (int): number of bootstrap replicates

    Returns:
        cov (np.array): the estimated covariance matrix, shape pxp.
        variances (np.array): the variance on each entry of cov, shape pxp.
    """
    # Initialization
    N = samp.shape[0]
    rndgen = np.random.default_rng(seed=seed)
    # Generate nboot samples of length fN with replacement,
    # compute the median of each
    boot_covs = []
    for i in range(nboot):
        # Uniform distributions for all samples with replacement.
        # Choice along axis 0, so choice of points (and not of dimensions!).
        boot_sample = rndgen.choice(samp, size=int(f*N), replace=True)
        boot_res = estimate_empirical_covariance(boot_sample, do_variance=False)
        boot_covs.append(boot_res[0])

    # Take the average covariance matrix, and compute the variance of each entry
    boot_covs = np.asarray(boot_covs)
    cov = np.mean(boot_covs, axis=0)
    variances = np.var(boot_covs, axis=0, ddof=1)  # normalize by N-ddof, unbiased.
    return cov, variances


def build_symmetric(flatmat):
    """ Build a symmetric matrix, assuming flatmat are the nonzero elements
    of the flattened upper triangular matrix (including the diagonal).

    Args:
        flatmat (np.ndarray): 1d (flat) array
    Returns:
        symmat (np.ndarray): 2d symmetric array
    """
    matsize = int(-0.5 + 0.5*np.sqrt(1 + 8*flatmat.size))
    symmat = np.zeros([matsize, matsize])
    symmat[np.triu_indices(matsize)] = flatmat
    low_ind = np.tril_indices(matsize, k=-1)
    symmat[low_ind] = np.triu(symmat, k=1).T[low_ind]  # Put the upper part in the lower one
    assert np.all(symmat == symmat.T)
    return symmat


def cholesky_variance(chol, n):
    """ Given a Cholesky decomposition chol = L_{ij} = T / \sqrt(n), where
    TT^T = S the sample covariance matrix, taken as a statistical estimator
    of \Psi, the Cholesky decomposition of the true covariance \Sigma,
    and a number of population sample points, compute the variance of each
    estimator element in the matrix. Returns a full, lower triangular matrix.

    Formula, adapted and corrected from Olkin 1985 (which made a mistake when
    reporting the variance of his unbiased estimators):
        Var[L_{ij}] = L_{ij}^2 v_j + \frac{1}{n-1} \sum_{k=j+1}^i L_{ik}^2
    where
        v_j = \frac{(n - j) - a_j^2}{n-1}
        a_i = \sqrt{2} \frac{ \Gamma{(n-j+1)/2} }{ \Gamma{(n-j)/2} }
    for i >= j; start with i=j, where the sum on the right is of course 0,
    then move to smaller and smaller j, reusing previous results in the sum.
    Store the partial sum as we decrease j, adding the current L_{ij}^2
    to it after computing its variance with the previous sum.

    Args:
        chol (np.ndarray): 2d array or matrix which is assumed to be
            lower triangular, positive definite.
        n (int): number of sample points used when chol was estimated.

    Returns:
        vari (np.ndarray): matrix of the same shape as chol giving
            the variance of each element in chol.
    """
    sqof2 = np.sqrt(2.0)
    if chol.shape[0] != chol.shape[1]:
        raise ValueError("chol should be a p x p matrix")
    vari = np.zeros(chol.shape)
    v_coefs = np.zeros(chol.shape[0])
    # Work row per row
    for i in range(chol.shape[0] - 1, -1, -1):
        running_row_sum = 0.0
        # Start on the diagonal, work towards earlier columns, down to 0th
        # adding 1 term below to the sum starting at j+1 each time j decreases
        for j in range(i, -1, -1):
            # If we are in the last row, compute the v_j and store them
            if i == chol.shape[0] - 1:
                a_j = sqof2 * np.exp(gammaln((n-j+1)/2.0) - gammaln((n-j)/2.0))
                v_coefs[j] = (n - j - a_j**2) / (n - 1)
            vari[i, j] = v_coefs[j] * chol[i, j]**2
            vari[i, j] += running_row_sum / (n - 1)
            running_row_sum += chol[i, j]**2
    return vari
