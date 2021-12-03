""" Small script to illustrate cytokine time courses (integrals) in 3D
and show the 2D plane of the latent space (its part in the 3D subspace).

To run this script, you need:
- Trained neural network weights in data/trained-networks/
- Processed cytokine time series in data/processed/

@author:frbourassa
June 11, 2020
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os
import seaborn as sns

# Can execute from any folder and it still works with this path modification
import os, sys
main_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if main_dir_path not in sys.path:
    sys.path.insert(0, main_dir_path)

from ltspcyt.scripts.neural_network import import_WT_output


## Function to read SI units
dico_units = {"k":1e3}
def read_tcn(text):
    value, units = [], []
    for a in text:
        if a.isnumeric():
            value.append(a)
        elif a.isalpha():
            units.append(a)

    # Put letters together and numbers together
    value = ''.join(value)
    units = ''.join(units)
    conc = float(value)

    # Read the order of magnitude
    units = units.replace("M", '')

    # If we encounter a problem here, put a value that is impossibly large
    if len(units) == 1:
        conc *= dico_units.get(units, 1e12)
    else:
        conc = 1e12

    return conc


def movie_cytokines_one_latent_plane(df, projmat, cytos, feat="integral"):
    """
    Args:
        df_data (pd.DataFrame): cytokine integrals for a dataset
        projmat (np.ndarray): 2d array, 2x5, each row is a vector
        cytos (list): string names of selected cytokines
        feat (str): either "integral", "concentration", or "derivative"
    """
    ### Initialization of the figure and the lines
    df2 = df.xs(feat, level="Feature", axis=1)
    # 3D plot with the three selected cytokines
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Determine the index of the selected cytokines
    available_cytos = ["IFNg", "IL-2", "IL-6", "IL-17A", "TNFa"]  # in the projection matrix
    index_cytos = [available_cytos.index(cy) for cy in cytos]

    # Prepare a list of peptides etc. that we want
    peps = df.index.get_level_values("Peptide").unique()
    peptide_order = ["N4", "Q4", "T4", "V4", "G4", "E1", "A2", "Y3", "A8", "Q7"]
    peps = [p for p in peptide_order if p in peps]
    concs = df.index.get_level_values("Concentration").unique()
    palette = sns.color_palette("colorblind")
    colors = {peps[i]:palette[i] for i in range(len(peps))}
    sizes = {concs[i]:1 + 0.3*i for i in range(len(concs))}
    times = df.index.get_level_values("Time").unique()

    # Plot first 2 hours of the trajectories, selecting the right cytokines
    # Save the lines and the keys, will need to iterate over them each time
    tstart = 2
    tsli = slice(times[0], times[tstart])
    last_pep = None
    lines, keys = [], []
    for pep in peps:
        for c in concs:
            try:
                x = df2.loc[(pep, c, tsli), cytos[0]].values
                y = df2.loc[(pep, c, tsli), cytos[1]].values
                z = df2.loc[(pep, c, tsli), cytos[2]].values
                lbl = pep if pep != last_pep else None
                li, = ax.plot(x, y, zs=z, color=colors[pep], lw=sizes[c],
                                label=lbl)
                last_pep = pep
            except KeyError:  # This combination dos not exist
                print("Not found")
                continue
            else:
                keys.append((pep, c))
                lines.append(li)

    ### Initialization of the movie: setting a few things on the plot
    # Define the point of view for the movie
    elevations = np.geomspace(3., 45., len(times)-tstart+1)
    azimuts = np.geomspace(-70, -160, len(times)-tstart+1)
    # Initialization function
    def finit():
        if feat == "integral":
            wrap = r"$\int \, \log_{{10}}({}) \, dt$"
        elif feat == "concentration":
            wrap = r"$\log_{{10}}({})$"
        else:
            wrap = r"$\frac{{d}}{{dt}} \, \log_{{10}}({})$"
        ax.set(xlabel=wrap.format(cytos[0].replace("-", "")),
               ylabel=wrap.format(cytos[1].replace("-", "")),
               zlabel=wrap.format(cytos[2].replace("-", "")))
        ax.view_init(elev=elevations[0], azim=azimuts[0])

        # Add the vectors on which we project
        props = {"arrowstyle":'->'}
        orig = [[0, 0]]*3
        xlim = [df2[cytos[0]].min(), df2[cytos[0]].max()]
        ylim = [df2[cytos[1]].min(), df2[cytos[1]].max()]
        zlim = [df2[cytos[2]].min(), df2[cytos[2]].max()]
        scale = -abs(min(xlim[1], ylim[1], zlim[1]))/2
        vec1, vec2 = projmat[0][index_cytos], projmat[1][index_cytos]
        vec1 /= np.sqrt(np.sum(vec1**2))
        vec2 /= np.sqrt(np.sum(vec2**2))
        tips = zip(vec1 * scale, vec2 * scale)
        ax.quiver(*orig, *tips, color="k", lw=3.)

        # Adjust boundaries to the arrows
        ax.set_xlim([min(xlim[0], vec1[0], vec2[0]), max(xlim[1], vec1[0], vec2[0])])
        ax.set_ylim([min(ylim[0], vec1[1], vec2[1]), max(ylim[1], vec1[1], vec2[1])])
        ax.set_zlim([min(zlim[0], vec1[2], vec2[2]), max(zlim[1], vec1[2], vec2[2])])
        ax.legend()
        return lines

    ### Update function: changing the lines' data to include one more time point
    # frames will be index of elements in the list of times
    def update(frame):
        tsli = slice(times[0], times[frame])
        for i in range(len(keys)):
            lines[i].set_xdata(df2.loc[(*keys[i], tsli), cytos[0]].values)
            lines[i].set_ydata(df2.loc[(*keys[i], tsli), cytos[1]].values)
            lines[i].set_3d_properties(df2.loc[(*keys[i], tsli), cytos[2]].values)

        # Update the point of view too...
        ax.view_init(elev=elevations[frame-tstart+1],
                     azim=azimuts[frame-tstart+1])
        return lines

    ### Animation
    ani = FuncAnimation(fig, update, frames=range(tstart, len(times)),
                    init_func=finit, blit=False, interval=100)
    return ani

def main_movie_one_plane():
    # Import data
    df_data = import_WT_output(folder=os.path.join(main_dir_path, "data", "processed/"))
    # Slice the desired values only
    df_data = df_data.loc[("PeptideComparison_3", "100k")]

    # Import projection matrix
    proj = np.load(os.path.join(main_dir_path, "data", "trained-networks",
                "mlp_input_weights-thomasRecommendedTraining.npy"))
    # Select cytokines
    chosen_cytokines = ["TNFa", "IL-2", "IL-17A"]  # ["IFNg", "IL-2", "TNFa"]
    # Produce the plot
    feat = "integral"
    anim = movie_cytokines_one_latent_plane(df_data, proj.T, chosen_cytokines, feat=feat)

    filename = os.path.join(main_dir_path, "figures", "latentspaces",
                    "3d_movie_{}_PeptideComparison3_{}.mp4".format(
                    "+".join(chosen_cytokines), feat))
    anim.save(filename, dpi=150)


    return 0

def cytokines_dataset_tcellnum_planes(df, projmat, cytos, feat="integral"):
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

    # Loop over datasets, change color each time
    dsets = df.index.get_level_values("Data").unique()
    tones = ["Reds", "Greys", "Blues", "Greens", "Oranges", "Purples", "pink"]
    for dset in dsets:
        df_dset = df.xs(dset, level="Data", axis=0)
        df_dset = df_dset.xs(feat, level="Feature", axis=1).unstack("Time")
        # Get the T cell numbers for this dataset
        tcnums = df_dset.index.get_level_values("TCellNumber").unique()
        # Sort in increasing T cell number
        tcnums = sorted(tcnums, key=read_tcn)
        tone = tones.pop(0)
        tones.insert(-1, tone)
        colors = sns.color_palette(tone, len(tcnums))
        for i, tcn in enumerate(tcnums):
            # Plot all lines for each tcn
            x = df_dset.loc[tcn, cytos[0]].values
            y = df_dset.loc[tcn, cytos[1]].values
            z = df_dset.loc[tcn, cytos[2]].values
            lbl = "{} {}".format(dset, tcn)
            # In 3D, need to plot one line at a time
            # (in 2D, can pass 2D array where each column is a curve)
            for r in range(x.shape[0]):
                if r == 0:
                    ax.plot(x[r], y[r], z[r], color=colors[i], label=lbl)
                else:
                    ax.plot(x[r], y[r], z[r], color=colors[i])
    if feat == "integral":
        wrap = r"$\int \, \log_{{10}}({}) \, dt$"
    elif feat == "concentration":
        wrap = r"$\log_{{10}}({})$"
    else:
        wrap = r"$\frac{{d}}{{dt}} \, \log_{{10}}({})$"
    ax.set(xlabel=wrap.format(cytos[0].replace("-", "")),
           ylabel=wrap.format(cytos[1].replace("-", "")),
           zlabel=wrap.format(cytos[2].replace("-", "")))
    ax.view_init(elev=-176, azim=-67)

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
    ax.legend(ncol=2, fontsize=8)

    # Possibly change angle of sight.
    return fig, ax

def main_effect_dataset_Tcellnum():
        # Import data
        df_data = import_WT_output(folder=os.path.join(main_dir_path, "data", "processed/"))
        # Slice the desired values only
        exp_selection = ["TCellNumber_1", "TCellNumber_3"]
        df_data = df_data.loc[exp_selection]

        # Import projection matrix
        proj = np.load(os.path.join(main_dir_path, "data", "trained-networks",
            "mlp_input_weights-thomasRecommendedTraining.npy"))
        # Select cytokines
        chosen_cytokines = ["IFNg", "IL-2", "TNFa"]  #["TNFa", "IL-2", "IL-17A"]
        # Produce the plot
        fig, ax = cytokines_dataset_tcellnum_planes(df_data, proj.T, chosen_cytokines)

        filename = os.path.join(main_dir_path, "figures", "supp",
            "3d_projection_{}_dataset2_tcellnum_effect.pdf".format(
            "+".join(chosen_cytokines)))
        fig.savefig(filename, transparent=True)
        plt.show()
        plt.close()

        return 0

if __name__ == "__main__":
    main_movie_one_plane()
    #main_effect_dataset_Tcellnum()
