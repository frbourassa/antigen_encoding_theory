#!/usr/bin/env python
# coding: utf-8
"""Script creating small cartoon 3D plots illustration the linear regression
reconstruction as fitting the best plane through 3D surfaces.

@author:frbourassa
February 8, 2021
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.transforms import Bbox
import matplotlib.colors as mpcolors
import scipy as sp
from scipy.interpolate import interp2d
from os.path import join as pjoin


def lsq_inverse(X, Y, y0=None):
    """ Compute the inverse projection matrix Q that minimizes
    the least-squares difference between the original data X and the
    reconstruction Q(Y-y0) from the projected points Y = PX + y0.
    The analytical solution shows that Q = XY^+, where Y^+ is
    the Moore-Penrose pseudo-inverse of Y, easily computed from
    the SVD of Y.

    Args:
        X (np.2darray): each column is a datapoint. size nxk, where n is the
            dimension of the initial space and k is the number of points.
        Y (np.2darray): each column is a projection. size mxk, where m is the
            dimension of the projection space and k is the number of points.
        y0 (np.1darray): possible offset vector added to points in Y
            after the projection.

    Returns:
        Q (np.2darray): a nxm inverse projection matrix that minimizes the
            squared error.
    """
    # Remove offsets
    Y2 = Y - y0 if (y0 is not None) else Y

    # Perform SVD
    u, s, vh = np.linalg.svd(Y2)

    # Compute pseudo-inverse
    k, m = Y2.shape[1], Y2.shape[0]
    inverter = np.vectorize(lambda x: 1/x if x !=0 else 0)
    splus = np.zeros([k, m])  # kxm matrix
    if k > m:
        splus[:m, :m] = np.diagflat(inverter(s))
    else:
        splus[:k, :k] = np.diagflat(inverter(s))
    Yplus = (vh.conj().T).dot(splus).dot(u.conj().T)

    # Compute and return Q
    return np.dot(X, Yplus)

def plane_eqn(l1, l2):
    # With 1d arrays of l1, l2, this will create a 2d array of points
    # where each column is a point.
    v1 = np.array([[0, np.sqrt(6/7), np.sqrt(1/7)]]).T
    v2 = np.array([[2*np.sqrt(18) + np.sqrt(2), -np.sqrt(2), np.sqrt(12)]]).T / (4*np.sqrt(7))

    return l1 * v1 + l2 * v2


def add_inset_3d(hostfig, rect, projection="2d", **kwargs):
    """ Add a 2D inset plot on a figure containing a 3D axis. """
    fig_coord = Bbox.from_bounds(*rect)
    if projection == "2d": projection = None
    axin = hostfig.add_axes(
        Bbox(fig_coord),
        projection=projection, **kwargs)
    return axin


def plot_recon_vs_data(origpts, reconpts, projpts, npts):
    # Choose a couple colors
    grey_background = list(mpcolors.to_rgba("grey"))
    grey_background[-1] = 0.2
    origcolor = "xkcd:dark grey"
    #reconcolor = plt.cm.viridis([206])[0]
    reconcolor = "xkcd:vibrant green"  # electric green, medium green, bright lime green, bright lime, vibrant green
    # Plot the result
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.set_size_inches(2.75, 2.75)

    ax.scatter(origpts[0], origpts[1], origpts[2],
                s=20, color=origcolor, label="Data")
    ax.plot_surface(reconpts[0].reshape(npts, npts),
                    reconpts[1].reshape(npts, npts),
                    reconpts[2].reshape(npts, npts),
                    color=reconcolor, alpha=0.4)
    ax.scatter(reconpts[0], reconpts[1], reconpts[2], s=9, color=reconcolor,
                label="Recon.")
    # ax.plot_surface(origpts[0].reshape(npts, npts),
    #             origpts[1].reshape(npts, npts),
    #             origpts[2].reshape(npts, npts),
    #             color=origcolor, alpha=0.6)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel(r"$x$", size=8, labelpad=-12)
    ax.set_ylabel(r"$y$", size=8, labelpad=-12)
    ax.set_zlabel(r"$z$", size=8, labelpad=-12)

    # Inset showing the projection.
    # BBox(left, bottom, width, height)
    axin = add_inset_3d(fig, [0.1, 0.6, 0.25, 0.25])
    axin.scatter(projpts[0], projpts[1], color=origcolor, s=5)
    axin.patch.set_facecolor(grey_background)
    #axin.patch.set_alpha(0.2)
    axin.set_xticks([])
    axin.set_yticks([])
    axin.set_xlabel(r"$N_1$", size=8)
    axin.set_ylabel(r"$N_2$", size=8)
    axin.set_title("Projection used \nto reconstruct", size=9)

    leg = ax.legend(framealpha=grey_background[-1], loc='upper right')
    leg.get_frame().set_facecolor(grey_background)
    # leg.get_frame().set_facecolor("none")
    leg.get_frame().set_linewidth(0.0)

    #ax.plot(projpts[0], projpts[1], marker="o",
    #    mfc="b", mec="b", zdir='x', zs=-1, ms=2, ls="none")
    #ax.text(-1, 0, 7, "Projection", zdir=(0, 1, 0))
    #ax.annotate("Projection", (0, 0, 80), xycoords="axes data")

    # Arrow to mark that we reconstruct from that projection.

    return [fig, ax, axin]


def recon_plane():
    npts = 8
    xx, yy = np.meshgrid(np.arange(npts), np.arange(npts))
    original_points = plane_eqn(xx.flatten(), yy.flatten())

    # Project (in xz plane) and try to reconstruct.
    proj_mat = np.array([[0, 1, 0], [0, 0, 1]])
    projected_points = np.dot(proj_mat, original_points)
    invproj_mat = lsq_inverse(original_points, projected_points)
    reconstructed_points = np.dot(invproj_mat, projected_points)

    fig, ax, axin = plot_recon_vs_data(original_points, reconstructed_points,
                                        projected_points, npts)
    fig.savefig(pjoin("figures", "supp", "3d_recon_example_plane.pdf"))
    plt.show()
    plt.close()


# What happens if the underlying manifold is a 2d surface, but not linear?
# It probably fails, no projection matrix can project on a curved surface.
# But it could be good enough. Let's see what happens with a paraboloid.
def paraboloid(x, y):
    # If x and y are 1d arrays, returns an array where each column is a point.
    return np.array([x, y, x**2 + y**2])

def recon_paraboloid():
    npts = 8
    xx, yy = np.meshgrid(np.arange(npts), np.arange(npts))
    original_points = paraboloid(xx.flatten(), yy.flatten())
    proj_mat = np.array([[0, 1, 0], [0, 0, 1]])
    projected_points = np.dot(proj_mat, original_points)
    invproj_mat = lsq_inverse(original_points, projected_points)
    reconstructed_points = np.dot(invproj_mat, projected_points)

    fig, ax, axin = plot_recon_vs_data(original_points, reconstructed_points,
                                        projected_points, npts)
    ax.view_init(azim=-45, elev=30)
    fig.savefig(pjoin("figures", "supp", "3d_recon_example_paraboloid.pdf"))
    plt.show()
    plt.close()

if __name__ == "__main__":
    # recon_plane()
    recon_paraboloid()
