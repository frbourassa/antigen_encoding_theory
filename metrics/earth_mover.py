"""
The Earth Mover's Distance is the minimal total work (distance*weight)
needed to rearrange the weight at points in x1 into the weight
configuration of points in x2, divided by the total weight transported.

It's a measure of how "easy" it is to morph one distribution into the other.
It comes from the classical problem (Monge, early 19th century) of
transporting one pile of dirt to another location, into some configuration.

For explanations and how it applies to distributions of biomarkers,
see https://doi.org/10.1371/journal.pone.0151859.

Warning: the following implementations of EMD have not been thoroughly tested. 

@author:frbourassa
March 18, 2021
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import networkx as nx


def emd_equal_samples(x1, x2):
    """
    Args:
        x1, x2 (np.2darrays): the data points in each of the two
            distributions to compare.
    Returns:
        emd (float): EMD, here equal to ratio of total transport distance
            (euclidean) divided by number of points, so average transport
            distance per point.

    This is a special case: we can use an integral assignment solution
    because all nodes have an equal weight of 1 initially.
    Then, the EMD just reduces to the average distance you need to move one
    point to optimally (least movement) rearrange points in x1 into the
    positions of points in x2.

    Quick solution for development purposes based on the following forum post:
    https://stackoverflow.com/questions/57562613/python-earth-mover-distance-of-2d-arrays
    """
    # Check the inputs, each row is a sample, each column a dimension
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    # Check dimensions and numbers of samples
    assert x1.shape[1] == x2.shape[1]
    # We are in a special case where we need the same number of samples
    # because we solve the transport problem for equal weights for simplicity
    nsamp = x1.shape[0]
    assert nsamp == x2.shape[0]

    # Build the pairwise distance matrix
    d_ij = cdist(x1, x2)

    # Solve the linear assignment problem (flow minimization with constraints)
    flow = linear_sum_assignment(d_ij)

    # Return the EMD: ratio of the total work divided by total distance (nsamp)
    emd = np.sum(d_ij[flow]) / nsamp
    return emd


def compute_emd(x1, x2, w1=None, w2=None):
    """ Actual EMD calculation using min flow algorithm for all the samples
    in x1 and x2, having weights in w1 and w2 (default is 1).

    Args:
        x1, x2 (np.ndarray): 2d arrays where each row is a data point
        w1, w2 (np.ndarray): 1d arrays giving the weight/mass of each
            point (each row) in x1 and x2, respectively.
    Returns:
        emd (float): the EMD, equal to the optimal (min) flow work
            divided by optimal total distance.

    Remark:
    We normalize the weights to make sure we have probability distributions.
    The EMD is then equivalent to the Mallow's distance. So it's not a pure
    EMD where we can also discard weight.

    """
    # Check the inputs, each row is a sample, each column a dimension
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    # Check dimensions and numbers of samples
    assert x1.shape[1] == x2.shape[1]

    # The weights of each node: amount of mass to export/receive.
    if w1 is None:
        w1 = np.ones(x1.shape[0])
    if w2 is None:
        w2 = np.ones(x2.shape[0])

    # Normalize the weights so the total is the same for the two distributions
    w1 = w1 / np.sum(w1)
    w2 = w2 / np.sum(w2)

    # networks prefers integer weights. Normalize sum to 1, round to n digits,
    # multiply by 10^n. Determine n as the log10 of the number of points + 1,
    # to make sure all weights don't drop to 0.
    scale_w1 = max(np.amin(w1), np.amax(w1)*1e-4)
    scale_w2 = max(np.amin(w2), np.amax(w2)*1e-4)
    scale_w = min(scale_w1, scale_w2) / 100  # Make this small enough for good rounding.
    w1 = (w1 / scale_w).astype(int)
    w2 = (w2 / scale_w).astype(int)

    # The rounding may make weights unequal. In that case, add weights to
    # the smallest one until they are equal
    sum1, sum2 = np.sum(w1), np.sum(w2)
    if sum1 < sum2:
        for i in range(sum2 - sum1):
            w1[i % w1.shape[0]] += 1
    elif sum1 > sum2:
        for i in range(sum1 - sum2):
            w2[i % w2.shape[0]] += 1

    # Compute euclidean distances, those are the weights of the EDGES
    # TODO: for large numbers of samples, should use nearest neighbor trees.
    # but at that point, one should reduce the distribution to a signature.
    matd12 = cdist(x1, x2)

    # Also need to round edge weights, ideally, for networkx
    scale_d = max(np.amax(matd12) * 1e-6, np.amin(matd12)/100)
    matd12 = (matd12/scale_d).astype(int)

    # Build the network. The first n1 nodes are initial (negative demand)
    g = nx.DiGraph()
    n1 = x1.shape[0]
    for i in range(n1):
        g.add_node(i, demand=-w1[i])
    for j in range(x2.shape[0]):
        g.add_node(n1 + j, demand=w2[j])

    # Add edges. The weight of each edge is its length.
    # Its capacity is the total weight in the initial point, to make sure
    # it does not transmit more mass than available in the initial point.
    for i in range(n1):
        for j in range(x2.shape[0]):
            g.add_edge(i, n1 + j, weight=matd12[i, j], capacity=w1[i])

    # Min flow solution. Dictionary of dictionaries keyed by nodes
    # such that flowDict[u][v] is the flow edge (u, v).
    try:
        min_flow = nx.algorithms.min_cost_flow(g)
    except nx.NetworkXUnfeasible as e:
        print(np.sum(w1), np.sum(w2))
    # Compute total work and total distance
    total_work = 0
    total_weight = 0
    for i in range(n1):
        for j in range(x2.shape[0]):
            f_ij = min_flow[i][n1 + j]  # Weight transported on that edge
            d_ij = matd12[i, j] * scale_d
            total_work += f_ij * d_ij
            total_weight += f_ij

    # Check that total_weight is indeed the total final weight w2 (or w1)
    if np.abs(total_weight - np.sum(w2)) > 1:
        print(total_weight)
        print(np.sum(w2))
        raise RuntimeError("You know nothing, FB")

    # Return EMD
    emd = total_work / total_weight
    return emd
