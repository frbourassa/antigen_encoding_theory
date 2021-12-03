""" Script to compute shifts in parameter space distribution when different
drugs are applied, using the Earth Mover's Distance (EMD) metric, the
Kolmogorov-Smirnov distance (K-S), or Kullback-Leibler divergence (KL_div).

This is a good measure to show that drugs create a significant proportion
of highly perturbed model parameter values, because it is not penalized
for overlap more than linearly (in the average). On the contrary, extreme high
values from points very far from the other distribution will influence
(just as extreme values skew a mean) because those faraway points need a lot
of transportation work. So, for a distribution that is partially strongly
shifted, like with our drugs, the EMD will be high.

Extra part here: we first apply PCA and keep the second axis only, which
corresponds to the direction perpendicular to the diagonal.

To run this notebook, you need:
- Processed cytokine time series in data/processed/
- Coefficients of a trained neural network in data/trained-network/, 
    for projection to latent space, and min-max values of training data. 

@author: frbourassa, soorajachar
March 19, 2021
"""
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import json
from time import perf_counter

# Can execute from any folder and it still works with this path modification
import os, sys
main_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if main_dir_path not in sys.path:
    sys.path.insert(0, main_dir_path)

from utils.statistics import principal_component_analysis
# All the metrics are available through appropriate_dist
from metrics.figure4_metrics import compute_distance_panel_PCA, appropriate_dist

from ltspcyt.scripts.latent_space import import_mutant_output
from ltspcyt.scripts.sigmoid_ballistic import return_param_and_fitted_latentspace_dfs


### Data processing functions
def rescale_per_dset(df):
    """ Code by Sooraj Achar, March 19, 2021 """
    means = df.mean(axis=0)
    variances = df.var(axis=0)
    standardizedDrugDfList = []
    for data in df.index.unique('Data'):
        dataDf = df.query('Data == @data')
        dataDf = pd.DataFrame(StandardScaler().fit_transform(dataDf.values),
                                index=dataDf.index, columns=dataDf.columns)
        standardizedDrugDfList.append(dataDf)
    df_full = pd.concat(standardizedDrugDfList)
    # Put back the overall mean value and variance in each column
    return df_full * np.sqrt(variances) + means


# To fit model parameters on drug experiments
def fit_params_drugs(data_folder=os.path.join("data", "processed"),
    model_choice="Constant velocity", regul_rate=1.0, **kwargs):
    """ Function to import pre-processed cytokine data of
    drug perturbation experiments and fit latent space model parameters on it.
        kwargs are passed directly to return_param_and_fitted_latentspace_dfs
    """
    # Import DrugPerturbation data
    df_mut = import_mutant_output(mutant="Drug", folder=data_folder)

    # Normalize data, project to latent space
    minmaxfile = os.path.join(main_dir_path, "data", "trained-networks", "min_max-thomasRecommendedTraining.hdf")
    df_min = pd.read_hdf(minmaxfile, key="df_min")
    df_max = pd.read_hdf(minmaxfile, key="df_max")
    projmat = np.load(os.path.join("data",
        "trained-networks", "mlp_input_weights-thomasRecommendedTraining.npy"))
    cytokines = df_min.index.get_level_values("Cytokine")
    times = np.arange(0, 73)
    df_mut = df_mut.unstack("Time").loc[:, ("integral", cytokines, times)].stack("Time")
    df_mut = (df_mut - df_min)/(df_max - df_min)
    df_proj = pd.DataFrame(np.dot(df_mut, projmat), index=df_mut.index,
                columns=["Node 1","Node 2"])

    # Curve fit of the chosen model
    print("Starting to fit parameters of model {}".format(model_choice))
    start_time = perf_counter()

    ret = return_param_and_fitted_latentspace_dfs(df_proj, model_choice,
        reg_rate=regul_rate, **kwargs)
    df_params, df_compare, df_hess, df_v2v1 = ret

    end_t = perf_counter()
    print("Time taken to fit: ", perf_counter() - start_time)
    del start_time, end_t

    nparameters = df_params.shape[1] // 2
    return df_params, df_compare, df_hess, df_v2v1


### Plotting function. Better visualization will be made
# for the main figures of the paper
def barplot_drug_panel(dst_array, dist_fct, drugnames, axis=1):
    # Sort such that peptides are at the end
    peptides = ["N4", "Q4", "T4", "V4", "G4", "E1", "A2", "Y3", "A8", "Q7"]
    # First sort drugs alphabetically
    names_plot = sorted(drugnames)
    pep_drugs = [a for a in names_plot if a not in peptides]
    sortkey = {a:i for i, a in enumerate(pep_drugs)}
    first_pep_idx = max(len(sortkey), 0)
    for pep in peptides:
        sortkey[pep] = len(sortkey)  # this increases by 1 each time
    arg_ord = np.argsort([sortkey[a] for a in names_plot])
    dst_plot = dst_array[arg_ord]
    names_plot = [names_plot[i] for i in arg_ord]

    # Plot
    fig1, ax1 = plt.subplots()
    barplot = ax1.bar(range(len(names_plot)), dst_plot)
    for i in range(first_pep_idx, len(names_plot)):
        barplot.patches[i].set_color("grey")  # color peptides apart
    ax1.axhline(0, color="grey", lw=0.8, ls="--", zorder=-1)
    ax1.set_xticks(range(len(names_plot)))
    ax1.set_xticklabels(names_plot, rotation=-90, fontsize=7)
    ax1.set_xlabel("Drug compared to no drug", size=9)
    dist_lbl = dist_fct.replace("_", " ")
    ax1.set_ylabel("{} on PC{} [-]".format(dist_lbl, axis), size=9)
    ax1.tick_params(which="both", labelsize=7)
    fig1.set_size_inches(6.75, 3.)
    fig1.tight_layout()
    return fig1, ax1


def main_drug_panel(df_p, dist_fct, add_sign=True):
    ## Reformat the dataframe
    to_drop = [a for a in df_p.index.get_level_values("Data").unique() if a.startswith("Lily")]
    df_p.drop(to_drop, level="Data", inplace=True)

    # Rename Dasatanib (typo) to Dasatinib and other small renames
    df_p = df_p.rename({"Dasatanib":"Dasatinib"}, level="Drug", axis=0)
    df_p = df_p.rename({"JAKi_AZD1480": "JAKi"}, level="Drug", axis=0)
    df_p = df_p.rename({'9':'9-12', '10':'9-12', '12':'9-12', '25':'24-25',
                        '24':'24-25'}, level='DrugAdditionTime')
    df_p = df_p.rename({'100nM':'1uM'}, level='Concentration')

    # Remove strange drugs
    to_drop = [a for a in df_p.index.get_level_values("Drug").unique() if "-Media" in a]
    df_p.drop(to_drop, level="Drug", inplace=True)

    # Remove DrugAdditionTime == 36, keep 100k T cells only
    df_p = df_p.xs("100k", level="TCellNumber", axis=0, drop_level=False)
    df_p = df_p.drop("36", level="DrugAdditionTime", axis=0)

    # Rescale per dataset so the parameter distributions match better.
    df_p = rescale_per_dset(df_p)

    # Fix dtype issue
    df_p = df_p.astype(np.float64)
    df_p = df_p.dropna(axis=0, how="any")
    assert np.amax(df_p.shape[0] - df_p.count(axis=0)) == 0

    # Select only the desired parameters.
    params_choice = ["v0", "theta"]
    df_p = df_p.loc[:, params_choice]

    # Compute PCA of the data once, to find the projection axis.
    pc_values, pc_axes, _ = principal_component_analysis(df_p.values)
    print("PCA values:", pc_values)
    print("PCA axes:", pc_axes)

    # Compute the distance of each drug compared to Null (for any peptide)
    # along each PC axis. Save all results.
    # Also, each peptide + no drug is compared to full distribution + no drug
    # as a way to see how each peptide deviates from the average distribution
    # This is performed by option add_intra of compute_distance_panel_PCA.
    dst_arrays, dst_columns = [], []
    for i in range(pc_axes.shape[1]):
        dst_array, all_drugs = compute_distance_panel_PCA(df_p,
                        pc_axes[:, i:i+1], ref_lbl="Null", lvl="Drug",
                        dist_fct=dist_fct, intra_pair=False, add_intra=True)
        dst_arrays.append(dst_array)
        dst_columns.append("{} along PC{}".format(dist_fct, i+1))

    # Concatenate the results into a DataFrame
    df_dist_pc = pd.DataFrame(np.stack(dst_arrays, axis=1),
            index=all_drugs, columns=dst_columns)
    df_dist_pc.index.name = "Drug"
    df_dist_pc.columns.name = dist_fct

    # Giving a sign to the distance based on the centroid displacement
    if add_sign:
        signDf = df_p.groupby("Drug").mean() - df_p.xs("Null", level="Drug").mean()
        signDf = pd.DataFrame(signDf.values.dot(pc_axes), index=signDf.index,
                    columns=["PC"+str(i+1) for i in range(len(dst_columns))])
        signDf = np.sign(signDf)
        signDf.loc["Null"] = 1
        signDf.columns.name = "PC"

        # Sign for the peptides
        pep_diffs = []
        peps_with_drugs = df_p.index.get_level_values("Peptide").unique()
        keynull = (slice(None), slice(None), "Null", slice(None), slice(None))
        for pep in peps_with_drugs:
            keypep = (slice(None), slice(None), "Null", slice(None), pep)
            pep_diffs.append(df_p.loc[keypep].mean() - df_p.loc[keynull].mean())
        pep_diffs = np.sign(np.asarray(pep_diffs))
        pep_signsDf = pd.DataFrame(pep_diffs.reshape(len(peps_with_drugs), len(dst_columns)),
                        columns=signDf.columns, index=pd.Index(peps_with_drugs, name="Drug"))
        signDf = signDf.append(pep_signsDf)
        signDf = signDf.rename({a:dist_fct+" along "+a for a in signDf.columns}, axis=1)
        df_dist_pc_signed = signDf * df_dist_pc
    else:
        df_dist_pc_signed = df_dist_pc

    # Save the distance DataFrame
    df_dist_pc_signed.to_hdf(os.path.join("results", "drugs",
        "df_drugs_{}_distances_alongPCs.hdf".format(dist_fct)), key="df")

    # Plot a barplot of the distance for each PC axis
    for i in range(pc_axes.shape[1]):
        fig, ax = barplot_drug_panel(df_dist_pc_signed.iloc[:, i].values,
             dist_fct, df_dist_pc_signed.index.get_level_values("Drug"), i+1)
        fig.savefig(os.path.join("figures", "drugs",
            "barplot_drugs_{}_distance_PC{}.pdf".format(dist_fct, i+1)))
        plt.show()
        plt.close()

    # Also show the best drugs on a scatter plot against null
    # Also show the PCA axes.
    ax = sns.scatterplot(data=df_p.reset_index(),
        x=params_choice[0], y=params_choice[1], hue="Peptide")
    vbase = df_p.mean(axis=0).values
    ax.annotate("PC1", xy=vbase, xytext=(vbase+pc_axes[:, 0]),
        arrowprops=dict(arrowstyle="<-", connectionstyle="arc3"))
    ax.annotate("PC2", xy=vbase, xytext=(vbase+pc_axes[:, 1]),
        arrowprops=dict(arrowstyle="<-", connectionstyle="arc3"))
    ax.set_aspect("equal")
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig = ax.figure
    fig.tight_layout()
    fig.savefig(os.path.join("figures", "drugs", "drugs_scatterplot_PCA.pdf"),
        transparent=True, bbox_extra_artists=(leg,), bbox_inches="tight")
    plt.show()
    plt.close()



if __name__ == "__main__":
    # Choose a distance metric
    distance_function = "EMD"  # Choices: ["EMD", "KL_div", "K-S"]

    # Import model parameter fits.
    try:
        df_params = pd.read_hdf(os.path.join("results", "drugs",
            "df_params_Constant_velocity_DrugPerturbations.hdf"))
    # Generate the model fits if they are unavailable
    except FileNotFoundError as e:
        df_params, _, _, _ = fit_params_drugs(data_folder=os.path.join("data",
            "processed"), model_choice="Constant velocity", regul_rate=1.0)
        df_params.to_hdf(os.path.join("results", "drugs",
            "df_params_Constant_velocity_DrugPerturbations.hdf"), key="df")

    # Compute distances along each PCA axis, save to disk, plot histograms.
    # Give a sign to the distance based on the directionality of the
    # shift along the PC vector.
    main_drug_panel(df_params, distance_function, add_sign=True)
