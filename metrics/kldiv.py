import numpy as np
from scipy.spatial import cKDTree as KDTree

# Code copied from the Internet for fast experimentation.
## Verdict: it's not as good as EMD
# Warning: few tests were run, either by the original author (DH) or me (FB)
# Author: David Huard, david.huard at gmail.com, May 2011
# https://mail.python.org/pipermail/scipy-user/2011-May/029521.html
# Code found on Github, posted by someone else:
# https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518
def KLdivergence(x, y):
    """Compute the Kullback-Leibler divergence between two multivariate samples.
    Parameters
    ----------
    x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
    y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.
    Returns
    -------
    out : float
    The estimated Kullback-Leibler divergence D(P||Q).
    References
    ----------
    Reference:
    Pérez-Cruz, F. Kullback-Leibler divergence estimation of
        continuous distributions IEEE International Symposium on Information
        Theory, 2008.
    """
    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n, d = x.shape
    m, dy = y.shape

    assert(d == dy), "x and y should have equal nb columns (dimension)"

    # Build a KD tree representation of the samples and find the nearest
    # neighbour of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(x, k=2, eps=.01, p=2)[0][:, 1]
    s = ytree.query(x, k=1, eps=.01, p=2)[0]

    # Modification by FB to remove problematic pairs (too close)
    logratio = np.log(r/s)
    logratio = logratio[np.isfinite(logratio)]

    # There is a mistake in the paper. In Eq. 14, the right side misses a
    # negative sign on the first term of the right hand side.
    return -logratio.sum() * d / n + np.log(m / (n - 1.))
