#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A module to process raw cytokine data.
Features:
- Flag and interpolate missing data
- log-transform and normalize data
- fit cubic splines
- extract features (integrals, concentrations & derivatives)

@author:tjrademaker
March 2020

based off a module by
@author:frbourassa
July 2019
"""
import os, sys
from ltspcyt.scripts.adapt_dataframes import set_standard_order

import numpy as np
import scipy
from scipy import interpolate
from scipy.stats import ks_2samp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def moving_average(points, kernelsize):
    """ Moving average filtering on the array of experimental points, averages over a block of size kernelsize.
    kernelsize should be an odd number; otherwise, the odd number below is used.
    The ith smoothed value, S_i, is
        $$ S_i = \frac{1}{kernelsize} \sum_{j = i-kernelsize//2}^{i + kernelsize//2} x_j $$
    Values at the boundary are smoothed with smaller and smaller kernels (up to size 1 for boundary values)

    Args:
        points (1darray): the experimental data points
        kernelsize (int): odd integer giving the total number of points summed

    Returns:
        smoothed (ndarray): the smoothed data points.
    """

    smoothed = np.zeros(points.shape)
    if kernelsize % 2 == 0:  # if an even number was given
        kernelsize -= 1
    w = kernelsize // 2  # width
    end = smoothed.shape[0]  # index of the last element

    # Smooth the middle points using slicing.
    smoothed[w:end - w] = points[w:end - w]
    for j in range(w):  # Add points around the middle one
        smoothed[w:-w] += points[w - j - 1:end - w - j - 1] + points[w + j + 1:end - w + j + 1]

        # Use the loop to treat the two points at a distance j from boundaries
        if j < w:
            smoothed[j] = np.sum(points[0:2*j + 1], axis=0) / (2*j + 1)
            smoothed[-j - 1] = np.sum(points[-2*j - 1:], axis=0)/(2*j + 1)

    # Normalize the middle points
    smoothed[w:end - w] = smoothed[w:end - w] / kernelsize

    return smoothed


def log_management(df, take, rescale, lod={}):
    """ Function to either take the log or normalize the concentrations
    Args:
        df (pd.DataFrame): the time series before taking the log. "Cytokine"
            should be the outermost level of the column MultiIndex.
        take (bool): whether to take the log or not
        lod (dict): lower limit of detection, in nM, of each cytokine.
    Returns:
        df_log (pd.DataFrame): the log or normalized series
    """
    df_log = pd.DataFrame(np.zeros(df.shape), index=df.index, columns=df.columns)
    # If log, make the smallest log 0 and the largest be
    # maxlog within each cytokine.
    # Else, we linearly rescale the concentrations between 0 and 10
    # Must proceed one cytokine at a time

    for cyt in df.columns.get_level_values("Cytokine").unique():
        df_cyt = df.xs(cyt, level="Cytokine", axis=1)
        min_conc = lod.get(cyt, df_cyt.values.min())

        if np.isnan(min_conc):
            min_conc = df_cyt.dropna().values.min()

        if take & rescale:
            max_conc = df_cyt.values.max()
            df_log.loc[:, cyt] = (np.log10(df_cyt.values) - np.log10(min_conc)) / (np.log10(max_conc) - np.log10(min_conc))
        elif take:
            df_log.loc[:, cyt] = np.log10(df_cyt.values) - np.log10(min_conc)
        else:
            df_log.loc[:, cyt] = df_cyt.values / max_conc * 10

    return df_log


def smoothing_data(df, kernelsize=3):
    """ Function to smooth all cytokine time series in the dataframe with a moving average filter.

    Args:
        data (pd.DataFrame): indexed with row levels (Peptide, Concentration) and column levels (Cytokine, Time)
        kernelsize (int): the number of points considered when averaging.
            Default: 3.

    Returns:
        smoothed (pd.DataFrame of UnivariateSpline): Spline objects, one per cytokine, ligand, and concentration triplet.
    """
    smoothed = pd.DataFrame(np.zeros(df.shape), index=df.index, columns=df.columns)
    for cyto in df.columns.get_level_values("Cytokine").unique():
        smt = moving_average(df[cyto].values.T, kernelsize=kernelsize)
        smoothed.loc[:, cyto] = smt.T
    return smoothed


def generate_splines(df, smoothed, rtol=1/2):#check_finite=True
    """ Function to prepare a DataFrame of splines objects, fitted on the
    inputted data. Same indexing as the raw data, but without time.

    Args:
        df (pd.DataFrame): the raw data, maybe after log management
        smoothed (pd.DataFrame): the smoothed data
        rtol (float): the fraction of the sum of squared residuals between
            raw and smoothed data used as a tolerance on spline fitting.

    Returns:
        spline_frame (pd.DataFrames): DataFrame of spline objects,
            one per cytokine per condition
    """
    # The experimental time points do not include time t = 0, of course, but we want
    # to integrate starting from t = 0. So, extrapolate to 0 by saying that
    # all cytokines are at their minimum value, which is zero.
    exp_times=df.columns.levels[1].to_list()
    inter_t = np.concatenate(([0], exp_times))
    # Create an empty DataFrame
    spline_frame = pd.DataFrame(None, index=df.index,
        columns=df.columns.get_level_values("Cytokine").unique(),
        dtype=object)
    for cyto in spline_frame.columns:
        for row in spline_frame.index:
            y = np.concatenate(([0],smoothed.loc[row, cyto]))
            r = np.concatenate(([0],df.loc[row, cyto]))
            tolerance = rtol * np.sum((y - r)**2)
            spl = scipy.interpolate.UnivariateSpline(inter_t, y, s=tolerance)
            spline_frame.loc[row, cyto] = spl
    return spline_frame


def lod_import(date, lod_folder=os.path.join("data", "LOD")):
    """ Function to import a LOD dictionary associated to the cytokine
    concentration file named cyto_file. Looks in lod_folder for a file
    containing the same experiment name than in cyto_file.

    LOD dictionary structure (Sooraj):  Each JSON file is a DataFrame that
    has 4 columns -- two for MFI and two for corresponding concentrations,
    min and max -- and 7 rows -- one for each cytokine.
    We are interested in the column ('Concentration', 'Lower').

    Args:
        date (str): the date of the cytokine experiment, format yyyymmdd
        lod_folder (str): path of the LOD folder, inclusively
    Returns:
        lower_bounds (dict): the dictionary containing the lower limit of
            detection for each cytokine (keys are cytokine names).
    """
    # Look for all LOD with the right date
    try:
        lod_file = [file for file in os.listdir(lod_folder)
                if ((date in file) & file.endswith(".json"))]
    except FileNotFoundError:
        lod_file = []

    if lod_file == []:
        print("Determined LOD from data for experiment {} ".format(date))
        return {}

    else:
        try:
            with open(os.path.join(lod_folder, lod_file[0]), "r") as handle:
                lod_df = pd.read_json(handle, orient="columns")
                print("Determined LOD from LOD file for experiment {}".format(date))
        except:
            print("Error while parsing LOD file {}; ".format(lod_file[0])
                    + "will use the minimum value of each cytokine instead.")
            return {}

        # Return only the lower bounds, in nM units
        lower_bounds = lod_df["('Concentration', 'Lower')"].to_dict()
        return lower_bounds

def treat_missing_data(df):
    """ Function to remove randomly or structurally missing datapoints, search for suspicious entries (zeros in all cytokines after having been nonzero).
    If found, set to NaN, then interpolate linearly

    Args:
        df (pd.DataFrame): ordered dataframe

    Returns:
        df (pd.DataFrame): ordered dataframe with zeros set to NaN
    """
    # Check for zeros (=minimum) per cytokine and time
    df_zero=(np.sum(df==df.min(),axis=1)==len(df.columns)).unstack("Time")

    # Logic: after having been nonzero cannot be zero in all cytokines at the same time
    remove_idx_time={}
    for time in range(1,len(df_zero.columns)):
        save_idx=[]
        for idx in range(len(df_zero)):
            if (not df_zero.iloc[idx,0:time].all()) & (df_zero.iloc[idx,time]):
                save_idx.append(idx)
        remove_idx_time[time]=save_idx

    # as a one liner
    # remove_idx_time={time:[idx for idx in range(len(df_zero)) if (not df_zero.iloc[idx,0:time].all()) & (df_zero.iloc[idx,time])] for time in range(1,len(df_zero.columns))}

    # Set missing data to NaN
    df_=df.copy()
    for k in remove_idx_time.keys():
        vals=remove_idx_time.get(k)
        if len(vals) == 0:
            continue
        for v in vals:
            df_.loc[tuple(list(df_zero.iloc[v,:].name)+[df_zero.columns[k]])] = np.nan

    # Interpolate NaNs linearly and return dataframe to desired shape
    df_=df_.interpolate(method="linear").unstack("Time")
    # For nonlinear interpolation methods applied to MultiIndex, see
    # https://stackoverflow.com/questions/32496062/how-can-i-interpolate-based-on-index-values-when-using-a-pandas-multiindex

    return df_


def extract_features(df_spline,max_time=72):
    """ Function to extract integrals, concentrations and derivatives from splines

    Args:
        df_spline (pd.DataFrame): dataframe of splines
        max_time (int): maximum time at which to extract features. Default = 72

    Returns:
        df (pd.DataFrame): dataframe with features
    """
    times=1+np.arange(max_time)
    df = pd.DataFrame(np.zeros((len(df_spline.index),len(times))), index=df_spline.index, columns=times)
    df.columns.name = "Time"
    df=pd.DataFrame(df.stack(level="Time"))
    df.columns = pd.MultiIndex.from_arrays([['integral'], ["IFNg"]], names=['Feature','Cytokine'])

    for cyto in df_spline.columns:
        df['integral',cyto] = np.array([[df_spline[cyto].iat[i].integral(0,time) for time in times] for i in range(len(df_spline.index))]).flatten()
        df['concentration',cyto] = np.array([df_spline[cyto].iat[i](times) for i in range(len(df_spline.index))]).flatten()
        df['derivative',cyto] = np.array([df_spline[cyto].iat[i].derivative()(times) for i in range(len(df_spline.index))]).flatten()

    return df


def update_integral_features(df_int):
    """ Function to make integrals monotonuous. Decreasing integrals are an
    artefact from the spline fitting procedure. Knots are fixed at start and
    end, which may cause the concentration to dip below the unphysical zero
    (in minimum of log transformed space), and thus integrals to decrease.

    Args:
        df_int(pd.DataFrame): dataframe with integral features (potentially nonmonotonuous)

    Returns:
        df_int (pd.DataFrame): dataframe with integral features without artificiality
    """
    df_int=df_int.unstack("Time").stack("Cytokine")
    for time in df_int.columns:
        df_int[time]-=np.nansum((df_int.diff(axis=1)[df_int.diff(axis=1)<0]).loc[:,np.arange(1,time+1)],axis=1)

    return df_int.stack("Time").unstack("Cytokine")


def find_date_in_name(f):
    potentials = [a for a in f.split("-") if a.isnumeric() and len(a) == 8]
    if len(potentials) > 1:
        date = f.split("-")[1]  # Hardcoded
    elif len(potentials) == 1:
        date = potentials[0]
    elif len(potentials) == 0:
        print("Could not find date in {}, returning 20000101".format(f))
        date = "20000101"
    return date


def process_file(folder,file, **kwargs):
    """ Function to process the raw cytokine concentrations time series:
    Find missing data points and linearly interpolate between them, take log, rescale and smooth with a moving average, interpolate with cubic splines, and extract features (integral, concentration & derivatives) at desired times
    Also tries to load limits of detection

    Args:
        data_file (str): path to the raw data file (a HDF5 pd.DataFrame)

    Keyword args:
        take_log (bool): True to take the log of the concentrations in the
            preprocessing, False if the networks have to deal with raw values.
            Default: True.
        rescale_max (bool): True: rescale concentrations by their maximum to
            account for experimental variability, False if we postpone
            normalization to a later stage.
            Default: False.
        smooth_size (int, default=3): number of points to consider when
            performing the moving average smoothing on the data points.
            In other words, width of the kernel.
        rtol_splines (float): tolerance for spline fitting: specify the
            fraction of the sum of squared residuals between the raw data
            and the data smoothed with a moving average that will be used
            as the total error tolerance in UnivariateSpline. Default: 1/2
        max_time (float): last time point to sample from splines.

    Returns:
        data (pd.DataFrame): the rearranged raw data, before processing.
        data_log (pd.DataFrame): the normalized log time series
        data_smooth (pd.DataFrame): log data after applying a moving average
        df (pd.DataFrame): processed data after extracting features from splines
    """
    # Processing-related keyword arguments
    take_log = kwargs.get("take_log", True)
    rescale_max = kwargs.get("rescale_max", False)
    smooth_size = kwargs.get("smooth_size", 3)
    rtol_splines = kwargs.get("rtol_splines", 1/2)
    max_time = kwargs.get("max_time", 72)

    # Import raw data
    data = pd.read_hdf(os.path.join(folder, file), key="df")

    # Put all timepoints for a given cytokine in continuous columns
    data = data.stack().unstack('Cytokine')

    # Check for randomly or structurally missing datapoints and interpolate between them
    data = treat_missing_data(data)

    # Import the limits of detection, if any
    # Get the path to the data/ folder.
    path_parts = os.path.normpath(folder).split(os.path.sep)[:-1]
    path_to_lod = os.path.join(*path_parts, "LOD")
    data_date = find_date_in_name(file)  # Date of the experiment.
    cytokine_lower_lod = lod_import(data_date, lod_folder=path_to_lod)

    # Take the log of the data if take_log, else normalize in linear scale
    data_log = log_management(data, take=take_log, rescale=rescale_max, lod=cytokine_lower_lod)

    # Smooth the data points before fitting splines for interpolation
    data_smooth = smoothing_data(data_log, kernelsize=smooth_size)

    # Fit cubic splines on the smoothed series
    spline_frame = generate_splines(data_log, data_smooth,rtol=rtol_splines)

    # Extract integral, concentration and derivative features from splines at set timepoints
    df = extract_features(spline_frame,max_time=max_time)

    # Update concentration and integral
    #TODO: speed of the code is limited by updating integrals. Optimizing this could make running time LIGHTNING FAST
    df[df.concentration<0]=0
    df["integral"] = update_integral_features(df.integral)

    # Return data in various stages of processing
    return [data, data_log, data_smooth, df]


### Special functions for extra noise filtering with K-S test
# This is not the usual processing pipeline but is sometimes necessary
def filter_null_batch(df, refnull, choice_cyto="IFNg", split_levels=[],
                      p_thresh=0.5, do_self=False, remove_cyto=[]):
    """ Function to compare time series, aggregated according to levels in
    split_levels, to a reference null peptide with a
    Kolmogorov-Smirnov 2-way test; if the test is conclusive  (p > p_thresh),
    that time series for that cytokine is set to zero exactly, since
    any deviation is assumed to be pure noise.

    Args:
        df (pd.DataFrame): log-treated time series dataframe. Columns should
            be cytokines and all other levels, including Time and Peptide,
            should be in the row index.
        refnull (str): label for the Peptide level which is the reference Null
        choice_cyto (str): cytokine on which to base the decision of whether or not
            to set cytokines in remove_cyto to zero
        split_levels (list): level(s) according to which time courses should be split.
            Default is none: all time courses for a peptide are aggregated.
        p_thresh (float): p-value threshold for the K-S test; time series
            with a larger p-value when compared to refnull will be
            considered Null too. Default: 0.5 (we need to be pretty sure)
        do_self (bool): if True, set the refnull peptide values to zero
            in the filtered dataframe.
        remove_cyto (list): which cytokines to remove if a group of time
            series is found to be similar to refnull.
    Returns:
        df_filt (pd.DataFrame): dataframe of the same shape as df,
            with cytokine time series found to be like refnull set to zero.
        filtered_sr (list of tuples): tuples giving the index key,
            cytokine, and p-value of the filtered out time series.
    """
    ## Find the index of the Peptide level
    try:
        pep_lvl_idx = df.index.names.index("Peptide")
    except ValueError:
        print("df does not have a Peptide level containing refnull in the row index; aborting filtering")
        return df

    ## Put time at the innermost index level to facilitate slicing
    try:
        df_filt = df.copy().stack("Time").unstack("Time")
    except KeyError:
        print("df does not have a Time level; aborting filtering")
        return df

    ## Prepare list of index keys over which we will slice.
    to_drop = list(df.index.names)
    idx_levels_kept = []
    for lvl in ["Peptide"]+split_levels:
        to_drop.remove(lvl)
        idx_levels_kept.append(df.index.names.index(lvl))
    idx_levels_kept.sort()
    # Tuples of labels that will be sliced to be compared to refnull peptide
    loop_index = df.index.droplevel(to_drop).unique()
    # Key in which we will update the sliced levels labels every iteration
    ky = np.asarray([slice(None)]*len(df.index.names))

    ## Loop over time series, and for the chosen cytokine, compare against refnull
    lod_cytos = df.stack("Time").min(axis=0)
    if remove_cyto == []:
        remove_cyto = df.stack("Time").columns.unique()
    filtered_sr = []
    for sr in loop_index:
        # Do not self-filter unless specified otherwise
        if refnull in sr and not do_self:
            continue
        # Update the labels of the levels to slice in the key ky
        ky[idx_levels_kept] = sr
        pep = ky[pep_lvl_idx]  # Save the current peptide, will need to put it back in ky
        timeser = df_filt.loc[tuple(ky)].stack("Time")
        ky[pep_lvl_idx] = refnull
        df_null = df.loc[tuple(ky)].stack("Time")

        # Check the cytokine of reference choice_cyto
        # p-value will be high if the timeser cumulative distribution
        # is larger at small x, i.e. timeser has smaller values than
        # refnull,  so the alternative (small p-value) is when timeser's
        # cdf is smaller at given x than for refnull.
        _, pval = ks_2samp(timeser[choice_cyto].values, df_null[choice_cyto].values,
                mode="exact", alternative="less")

        # If the (aggregated) timeseries is found to be similar to refnull
        if pval >= p_thresh:
            ky[pep_lvl_idx] = pep
            for cyt in remove_cyto:
                df_filt.loc[tuple(ky), cyt] = lod_cytos[cyt]
            filtered_sr.append((sr, pval))
    return df_filt, filtered_sr


def process_file_filter(folder,file, **kwargs):
    """ Function to process the raw cytokine concentrations time series:
    Find missing data points and linearly interpolate between them, take log, rescale and smooth with a moving average, interpolate with cubic splines, and extract features (integral, concentration & derivatives) at desired times
    Also tries to load limits of detection

    Args:
        data_file (str): path to the raw data file (a HDF5 pd.DataFrame)

    Keyword args:
        take_log (bool): True to take the log of the concentrations in the
            preprocessing, False if the networks have to deal with raw values.
            Default: True.
        rescale_max (bool): True: rescale concentrations by their maximum to
            account for experimental variability, False if we postpone
            normalization to a later stage.
            Default: False.
        smooth_size (int, default=3): number of points to consider when
            performing the moving average smoothing on the data points.
            In other words, width of the kernel.
        rtol_splines (float): tolerance for spline fitting: specify the
            fraction of the sum of squared residuals between the raw data
            and the data smoothed with a moving average that will be used
            as the total error tolerance in UnivariateSpline. Default: 1/2
        max_time (float): last time point to sample from splines.
        do_filter_null (bool): if True, filter (i.e. set to zero) time series
            that have a distribution similar to the null_reference peptide,
            as per a Kolmogorov-Smirnov test. Default: False
        null_reference (str): label of the peptide taken as the null reference
            for filtering. Default: "E1"
        choice_filter_cyto (str): cytokine according to which we filter.
            Default: None, then each single time series is compared to
            all aggregated null_reference time series.
        choice_remove_cyto (str): which cytokines to set to zero when a time
            series is found to be similar to the null_reference condition
            based on a comparison of choice_filter_cyto.
        split_filter_levels (str): levels according to which time series
            should be split during null peptide filtering; other levels
            are aggregated. Default: [].
        filter_pval (float): minimum K-S test p-value to filter a time series.
        remove_il17 (bool): if True, set IL-17A to zero in all time series.
        do_self_filter (bool): if True and filtering is on, null peptide
            will also have its choice_remove_cyto set to zero. Default: False.

    Returns:
        data (pd.DataFrame): the rearranged raw data, before processing.
        data_log (pd.DataFrame): the normalized log time series
        data_smooth (pd.DataFrame): log data after applying a moving average
        df (pd.DataFrame): processed data after extracting features from splines
        list_filt (list): (optional) list of the index of filtered time series.
            Only returned if return_list_filt keyword argument is set to True.
            The default is False, this is not returned.
    """
    # Processing-related keyword arguments
    take_log = kwargs.get("take_log", True)
    rescale_max = kwargs.get("rescale_max", False)
    smooth_size = kwargs.get("smooth_size", 3)
    rtol_splines = kwargs.get("rtol_splines", 1/2)
    max_time = kwargs.get("max_time", 72)
    do_filter_null = kwargs.get("do_filter_null", False)
    null_reference = kwargs.get("null_reference", "E1")
    filter_pval = kwargs.get("filter_pval", 0.5)
    choice_filter_cyto = kwargs.get("choice_filter_cyto", "IFNg")
    choice_remove_cyto = kwargs.get("choice_remove_cyto", ["IL-2"])
    split_filter_levels = kwargs.get("split_filter_levels", [])
    remove_il17 = kwargs.get("remove_il17", False)
    do_self_filter = kwargs.get("do_self_filter", False)
    return_list_filt = kwargs.get("return_list_filt", False)  #for diagnostics

    # Import raw data
    data = pd.read_hdf(os.path.join(folder, file), key="df")

    # Put all timepoints for a given cytokine in continuous columns
    data = data.stack().unstack('Cytokine')

    # Check for randomly or structurally missing datapoints and interpolate between them
    data = treat_missing_data(data)

    # Import the limits of detection, if any
    # Get the path to the data/ folder.
    path_parts = os.path.normpath(folder).split(os.path.sep)[:-1]
    path_to_lod = os.path.join(*path_parts, "LOD")
    data_date = find_date_in_name(file)  # Date of the experiment.
    cytokine_lower_lod = lod_import(data_date, lod_folder=path_to_lod)

    # Take the log of the data if take_log, else normalize in linear scale
    data_log = log_management(data, take=take_log, rescale=rescale_max, lod=cytokine_lower_lod)

    if remove_il17:
        data_log["IL-17A"] = 0.0

    # Optional: compare against a null reference to remove other null peptides
    if do_filter_null:
        data_log, list_filt = filter_null_batch(data_log,
            refnull=null_reference, choice_cyto=choice_filter_cyto,
            p_thresh=filter_pval, split_levels=split_filter_levels,
            remove_cyto=choice_remove_cyto, do_self=do_self_filter)
        print("Filtered {} null-like groups of levels {}".format(len(list_filt), split_filter_levels))
    else:
        list_filt = []

    # Smooth the data points before fitting splines for interpolation
    data_smooth = smoothing_data(data_log, kernelsize=smooth_size)

    # Fit cubic splines on the smoothed series
    spline_frame = generate_splines(data_log, data_smooth,rtol=rtol_splines)

    # Extract integral, concentration and derivative features from splines at set timepoints
    df = extract_features(spline_frame,max_time=max_time)

    # Update concentration and integral
    #TODO: speed of the code is limited by updating integrals. Optimizing this could make running time LIGHTNING FAST
    df[df.concentration<0]=0
    df["integral"] = update_integral_features(df.integral)

    # Return data in various stages of processing
    if return_list_filt:
        return [data, data_log, data_smooth, df, list_filt]
    else:
        return [data, data_log, data_smooth, df]
