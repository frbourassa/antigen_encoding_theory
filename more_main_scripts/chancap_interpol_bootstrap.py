""" Very specific script that regroups all channel capacity estimation
steps with the parameters appropriate for HighMI_3 in a function,
which can be called in parallel multiple time on different bootstrap samples

The __main__ block does the parallel computing and saves results.
It's easier to run multiprocessing from a script than a notebook; the latter
causes some bugs with more recent versions of multiprocessing.

To run this code, you need the same inputs as in the notebook
compute_channel_capacity_HighMI_3.ipynb.

@author:frbourassa
July 13, 2021
"""
# For multiprocessing
import multiprocessing

import numpy as np
import scipy as sp
import pandas as pd

# For the main function, which involves plotting
#import matplotlib.pyplot as plt
#import seaborn as sns
#plt.switch_backend("Agg")

import pickle, json
import sys, os  # to redirect terminal output to files
from time import perf_counter
from datetime import date
from psutil import cpu_count

# Can execute from any folder and it still works with this path modification
main_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if main_dir_path not in sys.path:
    sys.path.insert(0, main_dir_path)

from ltspcyt.scripts.process_raw_data import process_file, process_file_filter
from ltspcyt.scripts.sigmoid_ballistic import return_param_and_fitted_latentspace_dfs, compute_v2_v1
from utils.statistics import build_symmetric
from utils.distrib_interpolation import (eval_interpolated_means_covs, interpolate_params_vs_logec50,
                                         stats_per_levels, compute_cholesky_dataframe)
import utils.custom_pandas as custom_pd
import chancapmc.chancapmc as chancapmc


def capacity_from_latentspace_params_replicate_wrapper(seed, df, **kwargs):
    """ Wrapper around the function below to redirect stdout and stderr
    and close the files if an error arises
    """
    # Redirect terminal output to a different file
    stdout_backup = os.dup(1)
    stderr_backup = os.dup(2)
    sys.stdout.flush()
    fname = os.path.join(main_dir_path, "results", "capacity", "bootstrap",
                            "replicate_seed{}_terminal")
    f_out = open(fname.format(seed) + ".out", "w")
    f_err = open(fname.format(seed) + "_err.out", "w")
    sys.stdout = f_out
    sys.stderr = f_err
    # For the C output, need to overwrite the 1 and 2 file descriptors
    # see https://stackoverflow.com/questions/8804893/redirect-stdout-from-python-for-c-calls
    fd_out = os.open(fname.format(seed) + ".out", os.O_WRONLY)
    fd_err = os.open(fname.format(seed) + "_err.out", os.O_WRONLY)
    os.dup2(fd_out, 1)
    os.dup2(fd_err, 2)
    os.close(fd_out)
    os.close(fd_err)
    try:
        ret = capacity_from_latentspace_params_replicate(seed, df, **kwargs)
        return ret
    finally:
        f_out.close()
        f_err.close()
        sys.stdout = os.fdopen(stdout_backup, 'w')  # turn the file descriptor
        sys.stderr = os.fdopen(stderr_backup, 'w')  # into a file object
        print("Returning after seed {}".format(seed))



def capacity_from_latentspace_params_replicate(seed, df_proj, **kwargs):
    """
    Keyword arguments:
        time_scale (float)
        times (np.ndarray)
        choice_model (str)
        v2v1_mean_slope (float)
        regul_rate1 (np.ndarray or float)
        regul_rate2 (np.ndarray or float)
        slope1 (float)
        slope2 (float)
        params_correls1 (np.ndarray of ints)
        params_correls2 (np.ndarray of ints)
        pep_select1 (list of str)
        pep_select2 (list of str)
        tcn_fit (str)
        params_to_keep (list)
        n_inputs (int)
        reltol (float)
        save_inter (bool)
    """
    # Initialize the random generator to perturb the fit hyperparameters
    rgen = np.random.default_rng(seed=seed)

    ### Get some keyword arguments for model fitting common to all peptides
    time_scale = kwargs.get("time_scale", 20.0)
    times = kwargs.get("times", np.arange(1, 73))
    duration = np.amax(times)
    choice_model = kwargs.get("choice_model", "Sigmoid_freealpha")
    v2v1_mean_slope = kwargs.get("v2v1_mean_slope", 7.8)

    fit_vars={"Constant velocity":["v0", "t0", "theta", "vt"],
              "Constant force":["F", "t0", "theta", "vt"],
              "Sigmoid":["a0", "tau0", "theta", "v1", "gamma"],
              "Sigmoid_freealpha":["a0", "tau0", "theta", "v1", "alpha", "beta"]}
    nparameters = len(fit_vars[choice_model])

    # Special parameter boundaries for this dataset
    local_bounds_dict = {
        'Constant velocity':[(0, 0, -np.pi/3, 0), (6/20*time_scale, 5/20*time_scale, np.pi, 5/20*time_scale)],
        'Constant force': [(0, 0, -np.pi/3, 0), (6/20*time_scale, 5/20*time_scale, np.pi, 5/20*time_scale)],
        'Sigmoid':[(0, 0, -2*np.pi/3, -2/20*time_scale, time_scale/100),
                (8/20*time_scale, (duration + 100)/time_scale, np.pi/3, 2/20*time_scale, time_scale/2)],
        'Sigmoid_freealpha':[(0, 0, -2*np.pi/3, 0, time_scale/72, time_scale/72),
                            (8/20*time_scale, (duration + 100)/time_scale, np.pi/3, 2/20*time_scale,
                            time_scale/5, time_scale/2)]
    }
    special_bounds_dict = kwargs.get("special_bounds_dict", local_bounds_dict)


    ### Fit ballistic parameters for N4 down to T4
    pep_selection1 = kwargs.get("pep_select1", ["N4", "Q4", "T4", "A2", "Y3"])

    # Regularization
    regul_rate_local = 0.9*np.ones(nparameters)
    regul_rate_local[1] = 0.7  # reduce regularization on tau0 because there is a gap there now
    regul_rate1 = np.asarray(kwargs.get("regul_rate1", regul_rate_local))

    # Perturb the regularization rates
    if regul_rate1 is not None:
        regul_rate1 += 0.025*rgen.normal(size=regul_rate1.size)

    # Order of sigmoid free alpha parameters: a0, tau0, theta, v2, alpha, beta
    params_correls_local = np.asarray([
        [1, 0, int(1.9*10000), 0, int(0.05*10000)], # tau0 = 1.9*a0
        [2, 0, int(0.75*10000), int(-np.pi/2*10000), int(0.1*10000)]  # theta = 0.75*a0 - pi/2
    ], dtype=int).T
    params_correls1 = kwargs.get("params_correls1", params_correls_local)

    # Perturb the correlation coefficients and weights with uniform noise
    if params_correls1 is not None:
        nzs = 0.05*rgen.normal(size=2*params_correls1.shape[1])
        for i in range(params_correls1.shape[1]):
            params_correls1[2, i] += int(10000*nzs[2*i])  # Coefficient of proportionality
            params_correls1[4, i] += int(10000*nzs[2*i+1]) # Scale of the regularization term

    # Perturb the slope as well
    slope1 = kwargs.get("slope1", v2v1_mean_slope) + 0.025*rgen.normal()

    print("Starting parameter fitting for", pep_selection1)
    ret = return_param_and_fitted_latentspace_dfs(
            df_proj.loc[df_proj.index.isin(pep_selection1, level="Peptide")],
            choice_model, time_scale=time_scale, reg_rate=regul_rate1,
            reject_neg_slope=slope1, special_bounds_dict=special_bounds_dict,
            correls=params_correls1)
    df_params1, df_compare1, _, _ = ret


    ### Fit ballistic parameters for V4 down to E1
    pep_selection2 = kwargs.get("pep_select2", ["V4", "G4", "E1"])

    regul_rate_local = 0.6*np.ones(len(fit_vars[choice_model]))
    regul_rate_local[1] = 0.4  # tau0 is too constrained, it seems
    regul_rate2 = np.asarray(kwargs.get("regul_rate2", regul_rate_local))
    if regul_rate2 is not None:
        regul_rate2 += 0.025*rgen.normal(size=regul_rate2.size)

    # Enforce a tau0-a0 correlation pretty strongly
    params_correls_local = np.asarray([
        [1, 0, int(1.8*10000), 0, int(0.1*10000)], # tau0 = 1.8*a0
        [2, 0, int(0.5*10000), int(10000*-np.pi/2), int(0.1*10000)]  # theta = 0.5*a0 - 2
    ], dtype=int).T
    params_correls2 = kwargs.get("params_correls2", params_correls_local)

    # Perturb the correlation coefficients and weights with uniform noise
    if params_correls2 is not None:
        nzs = 0.05*rgen.normal(size=2*params_correls2.shape[1])
        for i in range(params_correls1.shape[1]):
            params_correls2[2, i] += int(10000*nzs[2*i])  # Coefficient of proportionality
            params_correls2[4, i] += int(10000*nzs[2*i+1]) # Scale of the regularization term

    # Perturb the slope as well
    slope2 = kwargs.get("slope2", v2v1_mean_slope) + 0.025*rgen.normal()

    # Use a different final slope, because for those peptides we can properly fit it
    print("Starting parameter fitting for", pep_selection2)
    ret2 = return_param_and_fitted_latentspace_dfs(
            df_proj.loc[df_proj.index.isin(pep_selection2, level="Peptide")],
            choice_model, time_scale=time_scale, reg_rate=regul_rate2,
            reject_neg_slope=slope2, special_bounds_dict=special_bounds_dict,
            correls=params_correls2)
    df_params2, df_compare2, _, _ = ret2

    ### Combine the results of the two sets of fits
    df_params = df_params1.append(df_params2)
    df_compare = df_compare1.append(df_compare2)

    del df_params1, df_params2, df_compare1, df_compare2

    ### Clean up a few outliers
    if choice_model.startswith("Constant"):
        df_params = df_params.loc[(df_params.index.isin(["V4"], level="Peptide")*df_params["tau0"]) < 1.25]
        df_params = df_params.loc[(df_params.index.isin(["V4"], level="Peptide")*df_params["theta"]) < 0.75]
        df_params = df_params.loc[(df_params.index.isin(["G4"], level="Peptide")*df_params["tau0"]) < 0.75]
        df_params = df_params.loc[(df_params.index.isin(["E1"], level="Peptide")*df_params["tau0"]) < 0.75]
        df_params = df_params.loc[(df_params.index.isin(["T4"], level="Peptide")*df_params["tau0"]) < 1.25]
        df_params = df_params.loc[(df_params.index.isin(["E1"], level="Peptide")*df_params["theta"]) < 0.5]
        df_params = df_params.loc[(df_params.index.isin(["G4"], level="Peptide")*df_params["theta"]) < 0.5]
    else:
        df_params = df_params.loc[np.logical_not(df_params.index.isin(["E1"], level="Peptide")*(df_params["theta"] > -np.pi/3))]
        df_params = df_params.loc[np.logical_not(df_params.index.isin(["T4"], level="Peptide")*(df_params["theta"] < -np.pi/2))]
        # Those G4 fits just make no sense, they don't represent the difference with V4
        df_params = df_params.loc[np.logical_not(df_params.index.isin(["G4"], level="Peptide")*(df_params["theta"] > -np.pi/3))]

    # Save details about what happened in that replicate
    # We don't need to save much more intermediate results as it's fairly easy
    # for the fitted parameters to check the multivariate normal fits and
    # the interpolation as a function of EC50.
    today = date.today().strftime("%d-%b-%Y").lower()
    if kwargs.get("save_inter", True):
        df_params.to_hdf(os.path.join(main_dir_path,
                    "results", "capacity", "bootstrap",
                    "df_params_highmi3_seed{}_{}.hdf".format(seed, today)),
                    key=str(seed), mode="a")
        params_correls1 = np.array([]) if params_correls1 is None else params_correls1
        params_correls2 = np.array([]) if params_correls2 is None else params_correls2
        regul_rate1 = np.array([]) if regul_rate1 is None else regul_rate1
        regul_rate2 = np.array([]) if regul_rate2 is None else regul_rate2
        kw_dict = {
            "slope1": slope1,
            "slope2": slope2,
            "params_correls1": params_correls1.tolist(),
            "params_correls2": params_correls2.tolist(),
            "regul_rate1": regul_rate1.tolist(),
            "regul_rate2": regul_rate2.tolist(),
        }
        with open(os.path.join(main_dir_path, "results", "capacity", "bootstrap",
                "hyperparameters_seed{}_{}.json".format(seed, today)),
                "w") as handle:
            json.dump(kw_dict, handle)


    ### Fit multivariate normal distributions in parameter space
    # Get some more kwargs relevant to distribution fitting
    params_to_keep = kwargs.get("params_to_keep", fit_vars[choice_model][:3])
    tcn_fit = kwargs.get("tcn_fit", "30k")
    levels_group = ["Peptide"]

    # This is where the multivariate normal distributions are fitted
    if tcn_fit == "all":
        ret = stats_per_levels(df_params, levels_groupby=levels_group, feats_keep=params_to_keep)
    else:
        ret = stats_per_levels(df_params.xs(tcn_fit, level="TCellNumber", axis=0),
                               levels_groupby=levels_group, feats_keep=params_to_keep)
    df_params_means, df_params_means_estim_vari, df_params_covs, df_params_covs_estim_vari, ser_npts = ret
    df_params_covs_estim_vari = np.abs(df_params_covs_estim_vari)

    # Also compute the Cholesky decomposition
    df_params_chol, df_params_chol_estim_vari = compute_cholesky_dataframe(df_params_covs, ser_npts)

    # Interpolate multivariate distributions as a function of log10(EC50)
    df_ec50s_refs = pd.read_json(os.path.join(main_dir_path, "data",
                                    "misc", "potencies_df_2021.json"))
    df_ec50s_refs.columns.name = "Reference"; df_ec50s_refs.index.name = "Peptide"
    ser_ec50s_avglog = np.log10(df_ec50s_refs).mean(axis=1)

    ser_splines_means = interpolate_params_vs_logec50(df_params_means,
                df_params_means_estim_vari, ser_ec50s_avglog, x_name="Peptide")
    ser_splines_chol = interpolate_params_vs_logec50(df_params_chol,
                df_params_chol_estim_vari, ser_ec50s_avglog, x_name="Peptide")
    ser_splines_covs = interpolate_params_vs_logec50(df_params_covs,
                df_params_covs_estim_vari, ser_ec50s_avglog, x_name="Peptide")
    ### Discretize the $log{EC_{50}}$ axis
    # n=25 seems an adequate number to extract all the information available
    n_inputs = kwargs.get("n_inputs", 25)
    min_logec50 = ser_ec50s_avglog.min()
    max_logec50 = ser_ec50s_avglog.max()
    bins_logec50 = np.linspace(min_logec50, max_logec50, n_inputs+1)
    # Take the midpoints of those bin separators
    sampled_logec50 = bins_logec50[:n_inputs] + np.diff(bins_logec50)/2

    # Compute means and covariance matrices at each EC50, from interpolation
    n_dims = len(df_params_means.columns)  # number of parameters
    meanmats, covmats, covmats_direct = eval_interpolated_means_covs(
        ser_splines_means, ser_splines_covs, ser_splines_chol, sampled_logec50,
        n_inputs, n_dims, epsil=1e-5)

    ### Run the Blahut-Arimoto algorithm
    seed_ba = seed*1241335 % 74137  # This seems to give pretty uniformly distributed numbers with seed = 342394+100*i
    reltol = kwargs.get("reltol", 0.01)

    start_t = perf_counter()
    res = chancapmc.ba_discretein_gaussout(meanmats, covmats, sampled_logec50, reltol, seed_ba)

    capacity_bits, optim_input_distrib = res
    capacity_error = capacity_bits * reltol

    print("Capacity = {} pm {}".format(capacity_bits, capacity_error))
    print("Optimal input distribution:", optim_input_distrib)

    run_duration = perf_counter() - start_t
    print("Time to converge: ", run_duration, "s")

    del start_t

    ### Save all the info on the run in a JSON file
    today = date.today().strftime("%d-%b-%Y").lower()
    run_info = {
        "date": today,
        "capacity_bits": capacity_bits,
        "input_values": list(sampled_logec50.astype(float)),
        "optim_input_distrib": list(optim_input_distrib.astype(float)),
        "n_inputs": n_inputs,
        "run_duration (s)": run_duration,
        "relative_tolerance": reltol,
        "absolute_error": capacity_error,
        "params": list(set(a for lbl in ser_splines_means.index
                              for a in lbl.split("*"))),
        "seed": seed_ba
    }
    folder = "capacity"
    suffix = "_HighMI_3"
    filename = os.path.join(main_dir_path, "results", "capacity", "bootstrap",
                "replicate_log{}_seed{}_{}.json".format(suffix, seed, today))
    with open(filename, "w") as hand:
        json.dump(run_info, hand)

    return run_info


def replace_time_series(df, lbl_to_replace, good_replicates, rndgen):
    """ The df should be indexed (Cytokine, Replicate, TCellNumber, Peptide, Concentration).
    Time should be in the columns.
    lbl_to_replace specifies slice(None), Replicate, ..., Concentration
    Warning: modifies the dataframe in place
    """
    choices = rndgen.choice(good_replicates, size=len(df.columns), replace=True)
    for i, tim in enumerate(df.columns):
        df.loc[lbl_to_replace, tim] = df.loc[(slice(None), choices[i], *lbl_to_replace[2:]), tim].values
    return 0


def main_bootstrap_channel_capacity(n_replicates, boot_frac=1.):
    """ Main function running n_replicates bootstrap replicates
    of ballistic parameter fitting plus channel capacity computation.
    Modify this function if the raw data used or the base hyperparameters
    have to change; otherwise, default hyperparameters as defined in
    capacity_from_latentspace_params_replicate are used.

    Returns:
        (tuple): avg_capacity, vari_capacity
        (tuple): avg_distrib, vari_distrib
    """
    ## Correct the raw data, removing faulty replicates
    # Import raw cytokine dataset. Concatenate back the nine separate files
    df_raw = {}
    for i in range(1, 10):
        df_raw[str(i)] = pd.read_hdf(os.path.join(main_dir_path, "data", "final",
                "cytokineConcentrationPickleFile-20210619-HighMI_3-{}-final.hdf".format(i)))
    df_raw = pd.concat(df_raw, names=["Replicate"])
    order_levels = np.asarray(df_raw.index.names)
    order_levels[:2] = order_levels[1::-1]
    df_raw = df_raw.reorder_levels(order_levels)
    del order_levels

    # Remove faulty replicates; for justification, see Jupyter notebook for HighMI_3
    # Basically, other stronger peptides have splashed in those wells

    # If you want to replace the E1 replicates instead of dropping them
    #randomgen = np.random.default_rng(seed=1349839)
    #for rep in ["2", "3", "5", "6", "8", "9"]:
    #    replace_time_series(df_raw, (slice(None), rep, "30k", "E1", "1uM"), ["1", "4", "7"], randomgen)

    # Drop faulty replicates for other peptides
    for cyto in df_raw.index.get_level_values("Cytokine").unique():
        for rep in ["2", "5", "8"]:
            df_raw = df_raw.drop((cyto, rep, "30k", "V4", "1nM"), axis=0)
        for rep in ["3", "6", "9"]:
            df_raw = df_raw.drop((cyto, rep, "30k", "G4", "1nM"), axis=0)
        for rep in ["2", "3", "5", "6", "8", "9"]:
            df_raw = df_raw.drop((cyto, rep, "30k", "E1", "1uM"), axis=0)

    # Write back to a file
    fname_corrected = "cytokineConcentrationPickleFile-20210619-HighMI_3_corrected-final.hdf"
    df_raw.to_hdf(os.path.join(main_dir_path, "data", "final",
                                        fname_corrected), key="df")

    ## Process the corrected data
    # Basically grouping all levels; TCellNumber not necessary because only one
    # value, and Peptide already done.
    # Use split_filter_levels=["Replicate", "Concentration"] if the E1
    # replicates were replaced instead of dropped, above.
    _, _, _, df_wt = process_file_filter(
                    folder=os.path.join(main_dir_path, "data", "final/"),
                    file=fname_corrected, do_filter_null=True, filter_pval=0.5,
                    null_reference="E1", choice_filter_cyto="IFNg",
                    choice_remove_cyto=["IL-2", "IL-17A", "TNFa", "IL-6"],
                    split_filter_levels=["Concentration"],
                    remove_il17=False, do_self_filter=True)

    # Write the corrected and processed file to hdf
    df_wt.to_hdf(os.path.join(main_dir_path, "data",
                                "HighMI_3_corrected.hdf"), key="df")

    ## Project to latent space and normalize
    peptides = ["N4", "Q4", "T4", "V4", "G4", "E1", "A2", "Y3"]
    concentrations = ["1uM", "100nM", "10nM", "1nM"]
    minmaxfile = os.path.join(main_dir_path, "data", "trained-networks",
                            "min_max-thomasRecommendedTraining.hdf")
    df_min = pd.read_hdf(minmaxfile, key="df_min")
    df_max = pd.read_hdf(minmaxfile, key="df_max")
    mlpcoefs = np.load(os.path.join(main_dir_path, "data", "trained-networks",
                "mlp_input_weights-thomasRecommendedTraining.npy"))

    cytokines = df_min.index.get_level_values("Cytokine")
    times = np.arange(1, 73)
    tcellnum = "30k"
    tcn_fit = "30k"
    folder = "capacity"
    suffix = "_HighMI"

    df_wt = pd.concat({"HighMI_3":df_wt}, names=["Data"])

    df = df_wt.unstack("Time").loc[:, ("integral", cytokines, times)].stack("Time")
    df = (df - df_min)/(df_max - df_min)
    df_proj_exp = pd.DataFrame(np.dot(df, mlpcoefs),
                    index=df.index, columns=["Node 1", "Node 2"])

    ## Import a v2/v1 slope from a dataset where IL-2 consumption was earlier
    df_highmi1 = pd.concat(
        {"HighMI_1-{}".format(i): pd.read_hdf(os.path.join(main_dir_path,
            "data", "processed", "HighMI_1-{}.hdf".format(i)))
        for i in range(1, 5)}, names=["Data"])
    df_highmi1 = df_highmi1.unstack("Time").loc[:, ("integral", cytokines, times)].stack("Time")
    df_highmi1 = (df_highmi1 - df_min)/(df_max - df_min)
    df_proj_highmi1 = pd.DataFrame(np.dot(df_highmi1, mlpcoefs),
                        index=df_highmi1.index, columns=["Node 1", "Node 2"])
    v2v1_mean_slope = compute_v2_v1(df_proj_highmi1, slope_type="mean", reject_neg_slope=True)

    ### Launch channel capacity bootstrap replicates
    n_points_per_pep = np.amax(df_proj_exp.iloc[:, 0]
        .xs(df_proj_exp.index.get_level_values("Time")[0], level="Time")
        .groupby("Peptide").count())
    n_sample_keep = int(boot_frac*n_points_per_pep)
    nb_cpus = cpu_count(logical=False)
    pool = multiprocessing.Pool(nb_cpus)
    list_return_objects = []  # to catch objects containing the returns
    seed = 342394
    for i in range(n_replicates):
        sd = seed + 100*i
        print("Launching seed", sd)
        # Create a bootstrap sample of the latent space data
        df_choice = (df_proj_exp.unstack("Time").groupby("Peptide")
            .sample(n=n_sample_keep, replace=True, random_state=sd)
            .stack("Time")).drop_duplicates()
        re = pool.apply_async(  # Launch a part of all runs
            capacity_from_latentspace_params_replicate_wrapper,
            args=(sd, df_proj_exp),
            kwds={"v2v1_mean_slope": v2v1_mean_slope, "save_inter":False,
                    "save_inter": True},
        )
        list_return_objects.append(re)

    # Get all the return values, they are ordered by seed
    list_returns = [p.get() for p in list_return_objects]
    pool.close()

    # Compute the average capacity and the average distribution
    all_capacities = np.asarray([a["capacity_bits"] for a in list_returns])
    all_distributions = np.asarray([a["optim_input_distrib"] for a in list_returns])
    avg_capacity = np.mean(all_capacities)
    # Variance of the bootstrap mean estimate: Young 2015,
    # Everything you wanted to know about data analysis..., eq. 2.67
    # Channel capacity is a highly non-linear function of
    # averages and covariances of the parameter space points
    # Bootstrapping more does not reduce the error on the average capacity estimator;
    # the error decreases with number of points, not n_boot.
    vari_capacity = np.var(all_capacities, ddof=0) * n_points_per_pep / (n_points_per_pep - 1)

    avg_distrib = np.mean(all_distributions, axis=0)
    vari_distrib = np.var(all_distributions, axis=0, ddof=0) * n_points_per_pep / (n_points_per_pep - 1)

    # Other arguments common to all replicate runs
    if len(list_returns) > 0:
        extra_keys = ["date", "input_values", "n_inputs", "relative_tolerance", "params"]
        moreargs = {k:list_returns[0][k] for k in extra_keys}

    return (avg_capacity, vari_capacity), (avg_distrib, vari_distrib), moreargs


if __name__ == "__main__":
    today = date.today().strftime("%d-%b-%Y").lower()
    multiprocessing.set_start_method('forkserver')
    n_cpu = cpu_count(logical=False)
    main_start_t = perf_counter()
    ret = main_bootstrap_channel_capacity(n_replicates=4*n_cpu, boot_frac=1.0)
    main_duration = perf_counter() - main_start_t
    print("Average capacity:", ret[0][0], "pm", np.sqrt(ret[0][1]), "bits")
    print("Average optimal distribution:", ret[1][0])
    print("Error on each term:", np.sqrt(ret[1][1]))
    ret_dict = {
        "average_capacity_bits": ret[0][0],
        "variance_capacity_bits": ret[0][1],
        "optimal_distribution": ret[1][0].tolist(),
        "variance_distribution": ret[1][1].tolist(),
        "run_duration (s)": main_duration
    }
    ret_dict.update(ret[2])
    with open(os.path.join(main_dir_path, "results", "capacity",
                "bootstrap_results_{}.json".format(today)), "w") as hand:
        json.dump(ret_dict, hand)

    # Combine all seed files
    full_run_dict = {}
    full_hyper_dict = {}
    folder = os.path.join(main_dir_path, "results", "capacity", "bootstrap")
    prefix = "all_bootstrap_replicates_"
    for fi in os.listdir(folder):
        seed_key = fi.split("_")[-2]
        if fi.startswith("df_params") and fi.endswith("{}.hdf".format(today)):
            dfp = pd.read_hdf(os.path.join(folder, fi))
            dfp.to_hdf(os.path.join(folder, prefix+"dfparams_"+today+".hdf"),
                mode="a", key=seed_key)
        elif fi.startswith("hyperp") and fi.endswith("{}.json".format(today)):
            with open(os.path.join(folder, fi), "r") as hand:
                full_hyper_dict[seed_key] = json.load(hand)
        elif fi.startswith("replicate_log") and fi.endswith("{}.json".format(today)):
            with open(os.path.join(folder, fi), "r") as hand:
                full_run_dict[seed_key] = json.load(hand)
        else:
            continue

    if len(full_hyper_dict) > 0:
        fi = prefix + "hyperparameters_{}.json".format(today)
        with open(os.path.join(folder, fi), "w") as hand:
            json.dump(full_hyper_dict, hand)
    if len(full_run_dict) > 0:
        fi = prefix + "runlogs_{}.json".format(today)
        with open(os.path.join(folder, fi), "w") as hand:
            json.dump(full_run_dict, hand)

    # Clean up individual files left behind
    for fi in os.listdir(folder):
        if fi.startswith("df_params") and fi.endswith("{}.hdf".format(today)):
            os.remove(os.path.join(folder, fi))
        elif fi.startswith("hyper") and fi.endswith("{}.json".format(today)):
            os.remove(os.path.join(folder, fi))
        elif fi.startswith("replicate_log") and fi.endswith("{}.json".format(today)):
            os.remove(os.path.join(folder, fi))
        else:
            continue
