#! /usr/bin/env python3
""" Python implementation of the mutual information estimator proposed by
Ross, 2014, "Mutual Information between Discrete and Continuous Data Sets",
PLOS One. Computes mutual information between a continuous multidimensional
variable and a discrete (categorical) variable, from an array of samples.

The formula for this estimator is
    $$ MI(X, Y) = psi(N) - <psi(N_x)> + psi(k) - <psi(m)> $$
where:
    - psi is the digamma function;
    - N is the number of points;
    - N_x is the number of points of category X=x;
    - k is the number of nearest-neighbors of the same category used to
        estimate the probability density;
    - m is the number of nearest-neighbors of any category within the ball
        extending up to the kth nearest-neighbor of the same category x;
    - <psi(N_x)> and <psi(m)> are averaged over samples of X (each category
        X=x is weighted by the number of occurences of x);

Speed improved by vectorizing some operations, using Scipy's cKDTree
for nearest-neighbor search, and using its multi-code capabilities.
Includes a direct translation in Python of the original Matlab code
provided by Ross 2014, for comparison (it is slower).

WARNING: limited testing of the code was carried out. It worked fine on the
test case provided below and for the authors' use cases, but results are not
guaranteed in other applications (especially with very high-dimensional data).

@authors: François Bourassa (frbourassa), Sooraj Achar (soorajachar)
Spring 2021
"""

import numpy as np
from scipy import special
from scipy.spatial import cKDTree
import psutil


def discrete_continuous_info_fast(d, c, k=3, base=np.e, eps=0):
    """
    Estimates mutual information between a discrete vector d and a continuous
    vector c (can be multidimensional) using nearest-neighbor statistics.
    Relatively fast Python implementation, using Scipy's cKDTree, of the
    estimator described by Ross, 2014, "Mutual Information between Discrete
    and Continuous Data Sets", PLOS One.
    Similar to the estimator described by Kraskov et al., 2004,
    "Estimating Mutual Information", PRE.

    Author of this implementation: Francois Bourassa (Github: frbourassa)

    Args:
        d (np.array): Array of discrete categories. Should be 1-dimensional
        c (np.array): Two-dimensional array or matrix of the continuous data.
            Dimensions should be (n samples x f features)
        k (int): Number of nearest neighbors for density estimation
        base (float): Logarithm base in which the MI is computed (default: e)
        eps (float): Relative tolerance on the radius up to which neighbors
            are included (default: 0, exact computation).

    Returns:
        float: Mutual information estimate
    """
    # Make sure d is an array to allow use of numpy functions
    if type(d) is list:
        d = np.asarray(d, dtype=type(d[0]))
    if d.ndim > 1:
        raise TypeError("d should be a 1d list or array")

    # First, prepare a list of categories according to the discrete symbols d
    numDimensions = c.shape[0]
    categories = list(np.unique(d))
    num_d_symbols = len(categories)

    # Build a KDTree of all points
    main_tree = cKDTree(c, leafsize=max(16, int(k*numDimensions/4)))

    # Number of workers for parallel processing, use half of them.
    n_workers = min(1, psutil.cpu_count() // 2)

    # Check that there are no exactly identical points
    identical_pairs = main_tree.query_pairs(r=0.0, eps=0.0, output_type="ndarray")
    # If any, perturb them slightly to avoid numerical instabilities
    dup_pts = np.unique(identical_pairs[:, 0])
    if identical_pairs.shape[0] > 0:
        # Average nn distance as a perturbation
        perturb = 1e-6*np.mean(main_tree.query(c, k=[2])[0])
        # Perturb the first point of each pair
        c[dup_pts, :] = c[dup_pts, :] + perturb*(np.random.random(size=(dup_pts.shape[0], c.shape[1])) - 0.5)
        # Update the tree
        main_tree = cKDTree(c, leafsize=max(16, int(k*numDimensions/4)))

    # Build an internal tree for each category
    m_tot = 0
    av_psi_Nd = 0
    psi_ks = 0
    for c_bin in range(num_d_symbols):
         # Slice elements in that category, save indices
        ii = (d == categories[c_bin]).nonzero()
        c_split = c[ii]
        numSamplesInBin = c_split.shape[0]
        one_k = min(k, numSamplesInBin-1)

        # For each point in the category, find the radius to its kth
        # nearest-neighbor, and count how many points of any category
        # are in that radius.
        if one_k > 0:
            # Build KDTree with leaf size one_k+2, since we won't need
            # to go much further
            categ_tree = cKDTree(c_split, leafsize=16)

            # Go to one_k+1 because self point is included as a neighbors.
            radii, indices = categ_tree.query(
                                c_split, [one_k+1], eps=eps, n_jobs=n_workers)

            # For each, count how many total points are within that distance.
            # Increase radii a little bit to make sure at least the one_k
            # neighbours in the category are found back
            # (float comparison issues in query in a different tree otherwise)
            m_points_all = main_tree.query_ball_point(
                                c_split, radii.ravel()*(1+1e-15), eps=eps,
                                n_jobs=n_workers, return_length=True)
            m_points_all -= 1  # the query includes the point itself

            # m_tot is \sum_i (psi(m_i))
            m_tot = m_tot + np.sum(special.psi(m_points_all))

        else:  # There was a single point in the category.
            m_tot = m_tot + special.psi(num_d_symbols*2)

        # Probability of each category given by its relative abundance in d
        p_d = numSamplesInBin/len(d)
        # Running estimates of the average digamma terms in the estimator
        av_psi_Nd += p_d*special.psi(p_d*len(d))
        psi_ks = psi_ks + p_d * special.psi(max(one_k, 1))
    # Computing the estimator
    f = special.psi(len(d)) - av_psi_Nd + psi_ks - m_tot/len(d)
    return f / np.log(base)


def discrete_continuous_info_ref(d, c, k=3, base=np.exp(1)):
    """
    Estimates the mutual information between a discrete vector 'd' and a
    continuous vector 'c' using nearest-neighbor statistics.
    Python translation of the estimator coded in Matlab
    Ross, 2014, https://doi.org/10.1371/journal.pone.0087357
    Similar to the estimator proposed by Kraskov et al., 2004,
    "Estimating Mutual Information", PRE.

    Author of the translation: Sooraj Achar (Github: soorajachar)

    Parameters:
        d (np.array): List of discrete categories. Should be 1 dimensional
        c (np.matrix): Matrix of continuous data.
            Dimensions should be n samples x f features
        k (int): Number of nearest neighbors
        base (float): Logarithm base in which the MI is computed (default: e)

    Returns:
        float: Mutual information estimate
    """
    #Make sure d is an array to allow use of numpy functions
    if type(d) is list:
        d = np.array(d, dtype=str)
    #Remove eventually
    if c.shape[1] < c.shape[0]:
        c = c.T
    first_symbol = []
    symbol_IDs = np.zeros(len(d))
    c_split = []
    cs_indices = []
    num_d_symbols = -1
    #First, bin the continuous data "c" according to the discrete symbols "d"
    numDimensions = c.shape[0]
    categories = list(np.unique(d))
    num_d_symbols = len(categories)
    for category in categories:
        ii = np.where(d == category)[0]
        cs_indices.append(ii)
        #Flip later
        c_split.append(c[:, ii])

    m_tot = 0
    av_psi_Nd = 0
    all_c_distances = np.zeros(len(d))
    psi_ks = 0
    for c_bin in range(num_d_symbols):
        numSamplesInBin = c_split[c_bin].shape[1]
        one_k = min(k, numSamplesInBin-1)
        if one_k > 0:
            c_distances = np.zeros([numSamplesInBin])
            for pivot in range(numSamplesInBin):
                # find the radius of our volume using only those samples with
                # the particular value of the discrete symbol 'd'
                for cv in range(numSamplesInBin):
                    vec_diff = (c_split[c_bin][:, cv]
                                    - c_split[c_bin][:, pivot])
                    vec_diff = np.reshape(vec_diff, (-1, c.shape[0])).T
                    c_distances[cv] = np.sqrt(np.dot(vec_diff.T, vec_diff))
                sorted_distances = np.sort(c_distances)
                eps_over_2 = sorted_distances[one_k]   # don't count pivot

                #count the number of total samples within our volume using all
                #  samples (all values of 'd')
                for cv in range(c.shape[1]):
                    vec_diff = c[:, cv] - c_split[c_bin][:, pivot]
                    vec_diff = np.reshape(vec_diff, (-1, c.shape[0])).T
                    all_c_distances[cv] = np.sqrt(np.dot(vec_diff.T, vec_diff))
                # Don't count pivot point
                neigh_all_categs = np.where(all_c_distances <= eps_over_2)
                m = max(len(all_c_distances[neigh_all_categs]) - 1, 0)
                m_tot = m_tot + special.psi(m)

        else:
            m_tot = m_tot + special.psi(num_d_symbols*2)

        p_d = numSamplesInBin/len(d)
        av_psi_Nd = av_psi_Nd + p_d*special.psi(p_d*len(d))
        psi_ks = psi_ks + p_d * special.psi(max(one_k, 1))

    f = special.psi(len(d)) - av_psi_Nd + psi_ks - m_tot/len(d)
    return f / np.log(base)


if __name__ == "__main__":
    # Some test case to make sure both approaches work the same way
    nsamp = 50
    data = np.zeros([nsamp*4, 2])
    target = np.zeros(nsamp*4)
    rndgen = np.random.default_rng(seed=12323452)
    for i in range(4):
        mean = np.asarray([i, i])
        cov = np.eye(2)*(i+1)*0.75
        data[nsamp*i:nsamp*(i+1)] = rndgen.multivariate_normal(
                                            mean=mean, cov=cov, size=nsamp)
        target[nsamp*i:nsamp*(i+1)] = i

    # Reference algorithm with for loops
    mi_ref = discrete_continuous_info_ref(target, data, k=3, base=2)
    print("Reference MI = ", mi_ref)

    # Fast, approximate algorithm using KDTrees and vectorized queries
    mi_fast = discrete_continuous_info_fast(target, data, k=3, base=2)
    msg = "Difference too large to be true: found "+str(mi_fast)
    assert abs(mi_ref - mi_fast) < 1e-10, msg
    print("Fast MI = ", mi_fast)
