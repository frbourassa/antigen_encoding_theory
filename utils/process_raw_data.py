#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A module to process raw cytokine data, with options to bypass
log transformation, smoothing and spline fitting, and time integration.

Features when full processing is applied:
- Flag and interpolate missing data
- log-transform and normalize data
- fit cubic splines
- extract features (integrals, concentrations & derivatives)

@author:tjrademaker
March 2020

based off a module by
@author:frbourassa
July 2019

modified October 2021 by:
@author: frbourassa
"""
import os
import numpy as np
import scipy
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def select_naive_data(df):
    """ Part of the import_WT_output function """
    naive_pairs={
        "ActivationType": "Naive",
        "Antibody": "None",
        "APC": "B6",
        "APCType": "Splenocyte",
        "CARConstruct":"None",
        "CAR_Antigen":"None",
        "Genotype": "WT",
        "IFNgPulseConcentration":"None",
        "TCellType": "OT1",
        "TLR_Agonist":"None",
        "TumorCellNumber":"0k",
        "DrugAdditionTime":36,
        "Drug":"Null",
        "ConditionType": "Control",
        "TCR": "OT1"
    }
    mask=[True] * len(df)
    for index_name in df.index.names:
        if index_name in naive_pairs.keys():
            mask=np.array(mask) & np.array([index == naive_pairs[index_name]
                            for index in df.index.get_level_values(index_name)])
            df=df.droplevel([index_name])
    return df[mask]

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


def log_management(df, take, rescale, remove=True, lod={}):
    """ Function to either take the log or normalize the concentrations
    Args:
        df (pd.DataFrame): the time series before taking the log. "Cytokine"
            should be the outermost level of the column MultiIndex.
        take (bool): whether to take the log or not
        remove (bool): whether to remove the min value or not. Always treated
            as True except if the log is not taken (i.e. take is False)
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
        elif rescale:
            df_log.loc[:, cyt] = df_cyt.values / max_conc * 10
        elif remove:  # It's only if no log or rescale is taken that we do not remove the min.
            df_log.loc[:, cyt] = df_cyt.values - min_conc
        else:  # No change
            df_log.loc[:, cyt] = df_cyt.values

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


def lod_import(date, path=""):
    """ Function to import a LOD dictionary associated to the cytokine
    concentration file named cyto_file. Looks in lod_folder for a file
    containing the same experiment name than in cyto_file.

    LOD dictionary structure (Sooraj):  Each pickle file is a dictionary that
    has 7 keys, one for each cytokine, each pointing to a list with 4 numbers.
        Number 1: Minimum Limit of Detection in GFI for cytokine
        Number 2: Maximum Limit of Detection in GFI for cytokine
        Number 3: Minimum Limit of Detection in Concentration for cytokine
        Number 4: Maximum Limit of Detection in Concentration for cytokine
    We are particularly interested in number 3. Numbers 1-2 can change if
    Sooraj performs dilutions.

    Args:
        cyto_file (str): the name of the cytokine data file.

    Returns:
        lower_bounds (dict): the dictionary containing the lower limit of
            detection for each cytokine (keys are cytokine names).
    """
    # Look for all LOD with the right date
    lod_file = [file for file in os.listdir(path+"data/LOD/") if ((date in file) & file.endswith(".pkl"))]

    if lod_file==[]:
        print("Will rescale with the minimum value of cytokines in the data, because it could not find the LOD file\n")
        return {}

    else:
        lod_dict=pd.read_pickle(path+"data/LOD/"+lod_file[0])

        # Return only the lower bounds, in nM units
        lower_bounds = {cy:a[2] for cy, a in lod_dict.items()}
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


def extract_feature(df_spline, feature, max_time=72):
    """ Function to extract integrals, concentrations and derivatives from splines

    Args:
        df_spline (pd.DataFrame): dataframe of splines
        feature (str): "integral" or "concentration"
        max_time (int): maximum time at which to extract features. Default = 72

    Returns:
        df (pd.DataFrame): dataframe with features
    """
    if feature not in ["concentration", "integral"]:
        raise ValueError("Feature {} not supported".format(feature))
    times=1+np.arange(max_time)
    df = pd.DataFrame(np.zeros((len(df_spline.index),len(times))), index=df_spline.index, columns=times)
    df.columns.name = "Time"
    df=pd.DataFrame(df.stack(level="Time"))
    df.columns = pd.MultiIndex.from_arrays([[feature], ["IFNg"]], names=['Feature','Cytokine'])

    # Compute concentration and integral features anyways
    # because the way we fix integrals depends on concentrations
    for cyto in df_spline.columns:
        df['integral',cyto] = np.array([[df_spline[cyto].iat[i].integral(0,time) for time in times] for i in range(len(df_spline.index))]).flatten()
        df['concentration',cyto] = np.array([df_spline[cyto].iat[i](times) for i in range(len(df_spline.index))]).flatten()

    # Fix potentially slightly negative concentrations due to spline artifacts
    # Also affects integrals, so need to do this whatever the desired feature
    df[df.concentration<0]=0
    # Only worth fixing integrals if the desired feature is integral
    if feature == "integral":
        df["integral"] = update_integral_features(df.integral)

    # Only return the relevant feature at the end
    return df.xs(feature, level="Feature", axis=1, drop_level=False)


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


def extract_empirical_feature(df_data, feature, max_time=72):
    """ Return concentrations at empirical time points, or compute
    integrals with the trapeze rule using only the empirical time points.
    """
    # Keep only times <= max_time
    df_feat = df_data.loc[:, df_data.columns.get_level_values("Time") <= max_time]
    if feature == "concentration":
        return pd.concat({feature:df_feat.stack("Time")},
                            names=["Feature"], axis=1)

    elif feature == "integral":
        # Integrate each time series. Put one per row for simpler slicing.
        df_feat = df_feat.stack("Cytokine")
        df_feat[0.0] = 0.0  # Add a column for zero
        df_feat = df_feat.sort_index(axis=1)  # Put zero column first
        assert df_feat.columns.name == "Time", "still something to stack"
        times = df_feat.columns.values.astype(np.float64)

        # Cumulative integrals with trapeze rule and unequal intervals
        df_int = 0.5*np.cumsum((df_feat.iloc[:, :-1].values + df_feat.iloc[:, 1:].values)
                                    * np.diff(times)[np.newaxis, :], axis=1)
        df_int = pd.DataFrame(df_int, index=df_feat.index, columns=pd.Index(times[1:], name="Time"))

        # Reformat to desired order
        df_int = df_int.unstack("Cytokine").stack("Time")
        df_int = pd.concat({feature:df_int}, names=["Feature"], axis=1)
        return df_int

def process_file_choices(folder, file, **kwargs):
    """ Function to process the raw cytokine concentrations time series:
    Find missing data points and linearly interpolate between them, take log,
    rescale and smooth with a moving average, interpolate with cubic splines,
    and extract features (integral, concentration & derivatives) at desired times
    Also tries to load limits of detection

    Args:
        data_file (str): path to the raw data file (a pickled pd.DataFrame)

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
    do_integrate = kwargs.get("do_integrate", True)
    do_smooth = kwargs.get("do_smooth", True)
    subtract_min = kwargs.get("subtract_min", True)

    ## Common processing steps
    # Import raw data
    data = pd.read_pickle(folder+file)
    # data = select_naive_data(data)
    # Original pipeline only selects data after processing

    # Put all timepoints for a given cytokine in continuous columns
    data = data.stack().unstack('Cytokine')

    # Check for randomly or structurally missing datapoints and interpolate between them
    data = treat_missing_data(data)

    # Import the limits of detection, if any
    cytokine_lower_lod = lod_import(file[32:40])

    ## Log or no log, rescale or just shift baseline to zero
    # Take the log of the data if take_log, else normalize in linear scale
    data_log = log_management(data, take=take_log, rescale=rescale_max, 
                              remove=subtract_min, lod=cytokine_lower_lod)

    ## Smoothing and interpolation, or not
    feat = "integral" if do_integrate else "concentration"
    if do_smooth:
        ## TODO: maybe change the smoothing procedure depending on whether
        # we have taken the log or not.
        # Smooth the data points before fitting splines for interpolation
        data_smooth = smoothing_data(data_log, kernelsize=smooth_size)
        # Fit cubic splines on the smoothed series
        spline_frame = generate_splines(data_log, data_smooth,rtol=rtol_splines)
        # Extract integral or concentration features from splines at set timepoints
        df = extract_feature(spline_frame, feat, max_time=max_time)

    # No smoothing: must use experimental samples only
    else:
        df = extract_empirical_feature(data_log, feat, max_time=max_time)

    # Return processed data
    return df
