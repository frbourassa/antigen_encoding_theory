import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import seaborn as sns

from utils.plotting_fits import read_conc

## Backup palette
palette_backup = plt.rcParams['axes.prop_cycle'].by_key()['color']
peptides_backup = ["N4", "Q4", "T4", "V4", "G4", "E1", "A2", "Y3", "A8", "Q7"]
colors_backup = {peptides_backup[i]:palette_backup[i] for i in range(len(peptides_backup))}

def cytokines_one_latent_plane(df, projmat, cytos, feat="integral"):
    """
    Args:
        df_data (pd.DataFrame): cytokine integrals for a dataset
        projmat (np.ndarray): 2d array, 2x5, each row is a vector
        cytos (list): string names of selected cytokines
        feat (str): either "integral", "concentration", or "derivative"
    """
    # 3D plot with the three selected cytokines
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Determine the index of the selected cytokines
    available_cytos = ["IFNg", "IL-2", "IL-6", "IL-17A", "TNFa"]  # in the projection matrix
    index_cytos = [available_cytos.index(cy) for cy in cytos]
    # Need to put $$ around names that should not be in math;
    # the label is inserted in the middle of an equation
    nice_cyto_labels = {"IFNg":r"$IFN-$\gamma", "TNFa":r"$TNF$", "IL-2":r"$IL-2$",
        "IL-6":r"$IL-6$", "IL-17A":r"$IL-17A$"}
    nice_cytos = [nice_cyto_labels.get(a, a) for a in cytos]

    # Prepare a list of peptides etc. that we want
    peps = df.index.get_level_values("Peptide").unique()
    peptide_order = ["N4", "A2", "Y3", "Q4", "T4", "V4", "Q7", "A8", "G4", "E1"]
    zorders = {peptide_order[i]:len(peptide_order)-i for i in range(len(peptide_order))}
    peps = [p for p in peptide_order if p in peps]
    concs = df.index.get_level_values("Concentration").unique()
    colors = colors_backup
    sizes = {concs[i]:1.5 + 0.3*i for i in range(len(concs))}

    # Plot trajectories, selecting right cytokines
    last_pep = None
    for i, pep in enumerate(peps):
        for c in concs:
            try:
                x = df.loc[(pep, c), (feat, cytos[0])].values
                y = df.loc[(pep, c), (feat, cytos[1])].values
                z = df.loc[(pep, c), (feat, cytos[2])].values
                lbl = pep if pep != last_pep else None
                ax.plot(x, y, zs=z, color=colors[pep], lw=sizes[c], label=lbl, zorder=zorders[pep])
                last_pep = pep
            except KeyError:  # This combination dos not exist
                print("Not found")
                continue

    if feat == "integral":
        wrap = r"$\int \, \log_{{10}}({}) \, dt$"
    elif feat == "concentration":
        wrap = r"$\log_{{10}}({})$"
    else:
        wrap = r"$\frac{{d}}{{dt}} \, \log_{{10}}({})$"
    ax.set_xlabel(wrap.format(nice_cytos[0]), size=9)
    ax.set_ylabel(wrap.format(nice_cytos[1]), size=9)
    ax.set_zlabel(wrap.format(nice_cytos[2]), size=9)
    ax.tick_params(which="both", labelsize=8)
    ax.view_init(elev=15., azim=240)

    # Add the vectors on which we project
    props = {"arrowstyle":'->'}
    orig = [[0, 0]]*3
    xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    scale = -abs(min(xlim[1], ylim[1], zlim[1]))/2
    vec1, vec2 = projmat[0][index_cytos], projmat[1][index_cytos]
    vec1 /= np.sqrt(np.sum(vec1**2))
    vec2 /= np.sqrt(np.sum(vec2**2))
    tips = zip(vec1 * scale, vec2 * scale)
    ax.quiver(*orig, *tips, color="k", lw=3.)

    # Adjust boundaries to the arrows
    xlim, ylim, zlim = list(xlim), list(ylim), list(zlim)
    ax.set_xlim([min(xlim[0], vec1[0], vec2[0]), max(xlim[1], vec1[0], vec2[0])])
    ax.set_xlim([min(ylim[0], vec1[1], vec2[1]), max(ylim[1], vec1[1], vec2[1])])
    ax.set_zlim([min(zlim[0], vec1[2], vec2[2]), max(zlim[1], vec1[2], vec2[2])])
    ax.legend(fontsize=8)

    # Possibly change angle of sight.
    return fig, ax


def cytokines_dataset_tcellstate_planes(df, projmat, cytos, hue_level="TCellNumber",
        feat="integral", init_view={}):
    """
    Args:
        df_data (pd.DataFrame): cytokine integrals for a dataset
        projmat (np.ndarray): 2d array, 2x5, each row is a vector
        cytos (list): string names of selected cytokines
        hue_level (str): dataframe level according to which color is selected
        feat (str): either "integral", "concentration", or "derivative"
        init_view (dict): "elev" and "azim"
    """
    # 3D plot with the three selected cytokines
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Determine the index of the selected cytokines
    available_cytos = ["IFNg", "IL-2", "IL-6", "IL-17A", "TNFa"]  # in the projection matrix
    index_cytos = [available_cytos.index(cy) for cy in cytos]
    nice_cyto_labels = {"IFNg":r"$IFN-$\gamma", "TNFa":r"$TNF$", "IL-2":r"$IL-2$",
        "IL-6":r"$IL-6$", "IL-17A":r"$IL-17A$"}
    nice_cytos = [nice_cyto_labels.get(a, a) for a in cytos]

    # Loop over datasets, change color each time
    dsets = df.index.get_level_values("Data").unique()
    tones = ["Greens", "Blues", "Greys", "Reds", "Oranges", "Purples", "pink"]
    dd = 0
    for dset in dsets:
        nice_dset_lbl = "Dset " + str(dd+1)
        dd += 1
        df_dset = df.xs(dset, level="Data", axis=0)
        df_dset = df_dset.xs(feat, level="Feature", axis=1).unstack("Time")
        # Get the T cell numbers for this dataset
        tcstates = df_dset.index.get_level_values(hue_level).unique()
        # Sort in increasing T cell number
        if hue_level == "TCellNumber":
            tcstates = sorted(tcstates, key=read_conc, reverse=True)
        tone = tones.pop(0)
        tones.insert(-1, tone)
        colors = sns.color_palette(tone, len(tcstates))[::-1]
        if len(dsets) == 1:  # single dataset, different T cell numbers.
            tone = "viridis"
            colors = sns.color_palette(tone, len(tcstates))
        if len(colors) > 4:  # Use two colors to see better.
            n1, n2 = len(tcstates) // 2 + len(tcstates) % 2, len(tcstates) // 2
            colors[:n1] = sns.color_palette(tone, n1)[::-1]
            tone = tones.pop(0)
            tones.insert(-1, tone)
            colors[n1:] = sns.color_palette(tone, n2)[::-1]
        for i, tcs in enumerate(tcstates):
            # Plot all lines for each tcn
            alph = 1 - 0.1*i
            x = df_dset.loc[tcs, cytos[0]].values
            y = df_dset.loc[tcs, cytos[1]].values
            z = df_dset.loc[tcs, cytos[2]].values
            if len(tcstates) > 1 and len(dsets) > 1:
                lbl = "{} {}".format(nice_dset_lbl, tcs)
            elif len(dsets) > 1:
                lbl = nice_dset_lbl
            elif len(tcstates) > 1:
                lbl = str(tcs)
            else:
                raise ValueError("Only one T cell number, one dataset: issue")
            # In 3D, need to plot one line at a time
            # (in 2D, can pass 2D array where each column is a curve)
            for r in range(x.shape[0]):
                if r == 0:
                    ax.plot(x[r], y[r], z[r], color=colors[i], label=lbl, alpha=alph)
                else:
                    ax.plot(x[r], y[r], z[r], color=colors[i], alpha=alph)

    if feat == "integral":
        wrap = r"$\int \, \log_{{10}}({}) \, dt$"
    elif feat == "concentration":
        wrap = r"$\log_{{10}}({})$"
    else:
        wrap = r"$\frac{{d}}{{dt}} \, \log_{{10}}({})$"
    ax.set_xlabel(wrap.format(nice_cytos[0]), size=9)
    ax.set_ylabel(wrap.format(nice_cytos[1]), size=9)
    ax.set_zlabel(wrap.format(nice_cytos[2]), size=9)
    ax.tick_params(which="both", labelsize=8)
    ax.view_init(elev=15., azim=240)
    if len(init_view.keys()) == 2:
        ax.view_init(**init_view)

    # Add the vectors on which we project
    props = {"arrowstyle":'->'}
    orig = [[0, 0]]*3
    xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    scale = -abs(min(xlim[1], ylim[1], zlim[1]))/2
    vec1, vec2 = projmat[0][index_cytos], projmat[1][index_cytos]
    vec1 /= np.sqrt(np.sum(vec1**2))
    vec2 /= np.sqrt(np.sum(vec2**2))
    tips = zip(vec1 * scale, vec2 * scale)
    #ax.quiver(*orig, *tips, color="k", lw=3.)

    # Adjust boundaries to the arrows
    xlim, ylim, zlim = list(xlim), list(ylim), list(zlim)
    ax.set_xlim([min(xlim[0], vec1[0], vec2[0]), max(xlim[1], vec1[0], vec2[0])])
    ax.set_xlim([min(ylim[0], vec1[1], vec2[1]), max(ylim[1], vec1[1], vec2[1])])
    ax.set_zlim([min(zlim[0], vec1[2], vec2[2]), max(zlim[1], vec1[2], vec2[2])])
    maplen = lambda x: len(x)
    if max(map(maplen, dsets)) < 12:
        ax.legend(ncol=2, fontsize=8)
    else:
        ax.legend(ncol=1, fontsize=8)

    # Possibly change angle of sight.
    return fig, ax
