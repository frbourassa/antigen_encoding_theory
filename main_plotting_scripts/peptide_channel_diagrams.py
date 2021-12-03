import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib as mpl
import os, sys
main_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if main_dir_path not in sys.path:
    sys.path.insert(0, main_dir_path)

import chancapmc.chancapmc as chancapmc

plt.rcParams["figure.figsize"] = (1.75, 1.5)
plt.rcParams["axes.labelsize"] = 9.
plt.rcParams["legend.fontsize"] = 7.
plt.rcParams["axes.labelpad"] = 1.
plt.rcParams["xtick.labelsize"] = 6.
plt.rcParams["ytick.labelsize"] = 6.

def cartoon_limited_peptide_distrib():
    peptides = ["N4", "Q4", "T4", "E1"]
    nconc_per_pep = [4, 4, 4, 1]
    ntot = float(sum(nconc_per_pep))
    probs = [1./ntot * p for p in nconc_per_pep]

    colors = sns.color_palette("colorblind", 4)
    fig, ax = plt.subplots()
    pad = 1./ntot*1.1
    ax.set_ylim((0, 1./ntot*max(nconc_per_pep) + pad))
    for i in range(4):
        ax.bar(i, probs[i], width=0.75, color=colors[i], align="center",
                edgecolor="k", linewidth=1.)
        ax.annotate(peptides[i]+"\n{} conc.".format(nconc_per_pep[i]),
                    xy=(i, probs[i] + pad*0.85), va="top", ha="center",
                    size=5)
    ax.invert_xaxis()
    ax.set_xticks(range(4))
    ax.set_xticklabels(peptides)
    ax.set_yticks([])
    ax.set_ylabel(r"$p_Q(q)$")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis="both", width=0.8, length=2.5)
    fig.tight_layout(h_pad=0.1, w_pad=0.1)
    fig.savefig(os.path.join("figures", "capacity",
                "cartoon_peptides_limited_distribution.pdf"),
                transparent=True, format="pdf")
    #plt.show()
    plt.close()


def cartoon_limited_param_distrib(nz=0.06):
    peptides = ["N4", "Q4", "T4", "E1"]
    # Generate fake gaussians in a 2D parameter plot (F vs t0)
    # Put some overlap between T4, V4 to explain how limited we are.
    means_F = np.asarray([0.25, 1.25, 1.5, 2.25])
    # Right half of a parabola
    def parabola(x):
        return 3. - (3. / means_F[-1]**2) * (x - means_F[-1])**2
    means_t0 = parabola(means_F)
    means_Ft0 = np.asarray(list(zip(means_F, means_t0)))
    sigmas_Ft0 = np.asarray([
                    np.asarray([[1., 0.], [0., 1.]])*nz*(1+i/8)
                for i in range(4)])

    # Sample those distributions
    rng = np.random.default_rng(seed=178434)
    nconc_per_pep = [4, 4, 4, 1]
    gaussian_samples = [
        rng.multivariate_normal(means_Ft0[3-i], sigmas_Ft0[3-i],
                                size=32*nconc_per_pep[i])
        for i in range(4)
    ]

    fig, ax = plt.subplots()
    colors = sns.color_palette("colorblind", 4)
    for i in range(4):
        ax.scatter(gaussian_samples[i][:, 0], gaussian_samples[i][:, 1], s=2.5**2,
        color=colors[i], alpha=0.7)

    # Add interpolation line and means
    f_range = np.linspace(means_F[0], means_F[-1], 100)
    t0_range = parabola(f_range)
    ax.plot(f_range, t0_range, lw=2., color="grey", alpha=0.8)
    for i in range(4):
        ax.plot(means_F[i], means_t0[i], marker="o", ls="none", ms=4,
            mfc=colors[3-i], mec="k", mew=0.8, alpha=0.8)

    ax.set_xlabel(r"$a_0$")
    ax.set_ylabel(r"$\tau_0$")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "capacity",
                "cartoon_parameters_limited_distribution.pdf"),
                transparent=True, format="pdf")
    #plt.show()
    plt.close()


def cartoon_full_param_distrib(do_optimize=True, nz=0.06):
    npep = 12
    peptides = np.arange(npep)
    # Generate fake gaussians in a 2D parameter plot (F vs t0)
    means_F = np.linspace(0., np.sqrt(2.5), npep)**2
    mxf = np.amax(means_F)
    # Right half of a parabola
    def parabola(x):
        return 3. - (3. / mxf**2) * (x - mxf)**2
    means_t0 = parabola(means_F)
    means_Ft0 = np.asarray(list(zip(means_F, means_t0)))
    sigmas_Ft0 = np.asarray([
                    np.asarray([[1., 0.], [0., 1.]])*nz*(1+f/8)
                for f in means_F])
    # Run the chancap algorithm on this ideal param distrib
    # Use this for the cartoon of the full peptide distribution.
    if do_optimize:
        res = chancapmc.ba_discretein_gaussout(means_Ft0,
                            sigmas_Ft0, peptides,0.01, 154320)
        optim_distrib = res[1]
        capacity = res[0]
        print("Capacity of cartoon example = {} bits".format(capacity))
    else:  # Just a fictitious quartic function, to illustrate edge effects.
        optim_distrib = (3*peptides/npep - 1.5)**4 - 1.5*(3*peptides/npep - 1.5)**2
        optim_distrib += np.abs(np.amin(optim_distrib))*2.5
        optim_distrib /= np.sum(optim_distrib)

    # Sample those distributions in proportional to the optimal distribution
    rng = np.random.default_rng(seed=178434)
    maxsamp = 48
    maxprob = np.amax(optim_distrib)
    gaussian_samples = [
        rng.multivariate_normal(means_Ft0[npep-1-i], sigmas_Ft0[npep-1-i],
            size=int(np.ceil(maxsamp / maxprob * optim_distrib[i])))
        for i in range(npep)
    ]

    fig, ax = plt.subplots()
    colors = sns.color_palette("magma", npep)[::-1]
    for i in range(npep):
        ax.scatter(gaussian_samples[i][:, 0], gaussian_samples[i][:, 1], s=2.5**2,
        color=colors[i], alpha=0.8)
    # Add interpolation line
    f_range = np.linspace(means_F[0], means_F[-1], 100)
    t0_range = parabola(f_range)
    ax.plot(f_range, t0_range, lw=2., color="grey", alpha=0.8)
    for i in range(npep):
        ax.plot(means_F[i], means_t0[i], marker="o", ls="none", ms=4,
            mfc=colors[npep-i-1], mec="k", mew=0.8)

    ax.set_xlabel(r"$a_0$")
    ax.set_ylabel(r"$\tau_0$")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    tag = "optim" if do_optimize else "no-optim"
    fig.savefig(os.path.join("figures", "capacity",
                "cartoon_parameters_full_distribution_{}.pdf".format(tag)),
                transparent=True, format="pdf")
    #plt.show()
    plt.close()
    return optim_distrib


def cartoon_full_peptide_distrib(optim_probs, do_optimize=True):
    npep = len(optim_probs)
    peptides = np.arange(npep)

    colors = sns.color_palette("magma", npep)[::-1]
    fig, ax = plt.subplots()

    ax.bar(peptides, optim_probs, width=1., color=colors, align="center",
            edgecolor="k", linewidth=1.)
    ax.invert_xaxis()  # Put E1 to the left.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"$\log_{10}$EC$_{50}$")
    ax.set_ylabel(r"$p_Q(q)$")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis="both", width=0.8, length=2.5)
    fig.tight_layout()
    tag = "optim" if do_optimize else "no-optim"
    fig.savefig(os.path.join("figures", "capacity",
                "cartoon_peptides_full_distribution_{}.pdf".format(tag)),
                transparent=True, format="pdf")
    #plt.show()
    plt.close()


def cartoon_parameter_interpolation(nz=0.06):
    fig = plt.figure()
    gs = fig.add_gridspec(2, 3)
    fig.set_size_inches(3.8, 1.60)
    ax_mF = fig.add_subplot(gs[0, 0])
    ax_mt = fig.add_subplot(gs[1, 0])
    axes_chl = [[fig.add_subplot(gs[i, j+1]) for j in range(2)] for i in range(2)]
    fs = 7

    npep = 12
    ec50_axis = np.arange(0, npep-1+0.05, 0.1)
    slope_F, intercept_F = (2.25-0.25)/(npep-1), 0.25
    # Where to locate data and interpolated peptides
    ec50_peps = (np.asarray([0.25, 1.25, 1.5, 2.25]) - intercept_F)/slope_F
    ec50_interp_peps = np.arange(0, npep)
    mxf = 2.25
    # Interpolation functions
    def F_interp(ec):  # Linear
        return slope_F * ec + intercept_F
    def t_interp(ec):
        # Compute F
        f = F_interp(ec)
        # Return parabola
        return 3. - (3. / mxf**2) * (f - mxf)**2
    # NB: we interpolate Cholesky elements.
    # Sqrt of a diagonal matrix: sqrt of diagonal terms.
    def chlmats_interp(ec):
        # Compute f
        f = F_interp(ec)
        if not isinstance(f, np.ndarray):
            f = np.asarray(f)
        # Use same covariance matrix interpolation as before. Cholesky: sqrt
        return np.asarray([np.asarray([[1., 0.], [0., 1.]])*nz*np.sqrt(1+fi/8)
                    for fi in f])

    # Now, put all that in very small plots
    pep_colors = sns.color_palette("colorblind", 4)[::-1]
    allpep_colors = sns.color_palette("magma", npep)
    interp_line_props = dict(lw=2., color="grey", alpha=0.8, zorder=0)
    allpep_mark_props = dict(c=allpep_colors, s=2.5**2, zorder=1)
    pep_mark_props = dict(c=pep_colors, s=4.5**2, edgecolors="k", zorder=2)

    # F vs EC50
    ax_mF.plot(ec50_axis, F_interp(ec50_axis), **interp_line_props)
    ax_mF.scatter(ec50_interp_peps, F_interp(ec50_interp_peps),
        **allpep_mark_props)
    ax_mF.scatter(ec50_peps, F_interp(ec50_peps), **pep_mark_props)
    ax_mF.set_xticks([])
    ax_mF.set_yticks([])
    ax_mF.set_ylabel("$a_0$", size=fs)

    # t0 vs EC50
    ax_mt.plot(ec50_axis, t_interp(ec50_axis), **interp_line_props)
    ax_mt.scatter(ec50_interp_peps, t_interp(ec50_interp_peps),
        **allpep_mark_props)
    ax_mt.scatter(ec50_peps, t_interp(ec50_peps), **pep_mark_props)
    ax_mt.set_xticks([])
    ax_mt.set_yticks([])
    ax_mt.set_xlabel(r"$\log_{10}(EC_{50})$", size=fs)
    ax_mt.set_ylabel(r"$\tau_0$", size=fs)

    # Cholesky elements: add some noise to the data points
    ecaxis_chl_mats = chlmats_interp(ec50_axis)
    peps_chl_mats = chlmats_interp(ec50_peps)
    allpeps_chl_mats = chlmats_interp(ec50_interp_peps)
    params_lbls = [r"$a_0$", r"$\tau_0$"]
    rndgen = np.random.default_rng(seed=121398)
    noise_mgn = 0.005
    for i in range(2):
        for j in range(2):
            # Labeling first
            chol_lbl = "Chol(" + params_lbls[i] + "," + params_lbls[j] +")"
            axes_chl[i][j].set_yticks([])
            axes_chl[i][j].set_xticks([])
            axes_chl[i][j].set_ylabel(chol_lbl, size=fs)
            # This Cholesky element is zero
            if j > i:
                axes_chl[i][j].annotate(
                    "0: Cholesky\n matrix is\nlower\ntriangular",
                    xy=(0.5, 0.5), va="center", fontsize=fs,
                    xycoords="axes fraction", ha="center")
                continue
            axes_chl[i][j].plot(ec50_axis, ecaxis_chl_mats[:, i, j],
                **interp_line_props)
            # Otherwise, continue plotting interpolation
            axes_chl[i][j].scatter(ec50_interp_peps, allpeps_chl_mats[:, i, j],
                **allpep_mark_props)
            # Add error bars and noise (re-center the small sample)
            nz = rndgen.normal(size=ec50_peps.size)*noise_mgn
            mat_peps =  peps_chl_mats[:, i, j] + (nz - nz.mean())
            axes_chl[i][j].errorbar(ec50_peps, mat_peps, yerr=noise_mgn*2,
                fmt="none", ls="none", elinewidth=1., ecolor="k")
            axes_chl[i][j].scatter(ec50_peps, mat_peps, **pep_mark_props)
            if i == 1:
                axes_chl[i][j].set_xlabel(r"$\log_{10}(EC_{50})$", size=fs)
    fig.tight_layout(h_pad=0.3, w_pad=0.3)
    fig.savefig(os.path.join("figures", "capacity",
        "cartoon_ballistic_params_distribs_interpolation.pdf"),
        transparent=True, bbox_inches="tight")
    #plt.show()
    plt.close()

if __name__ == "__main__":
    cartoon_limited_peptide_distrib()
    cartoon_limited_param_distrib()
    do_optimize = False
    optim_dist = cartoon_full_param_distrib(do_optimize=do_optimize)
    cartoon_full_peptide_distrib(optim_dist, do_optimize=do_optimize)
    cartoon_parameter_interpolation()
