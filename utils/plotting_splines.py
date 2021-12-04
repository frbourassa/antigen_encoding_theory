""" Utility functions to make supplementary figures about data processing.

@author:frbourassa
"""
import os
import pandas as pd

from ltspcyt.scripts.process_raw_data import (treat_missing_data,
    lod_import, log_management, smoothing_data, generate_splines)


## Utility functions used in plotting scripts about data processing
def nicer_name(fname):
    """ Extract a nice name for the Series from the raw file path.

    Args:
        fname (str): the file path
    Returns:
        (str): a nicer, shorter name.
    """
    folderlst = os.path.split(fname)
    fname = folderlst[-1]
    fragments = fname.split(sep="-")
    # Search for the date; the fragment after is the good one.
    idx = 0
    for frag in fragments:
        idx += 1
        if frag.isnumeric():
            break  # stop looping; the next one is the good one
    try:
        good_name = fragments[idx]
    except IndexError:
        print("No date was found in the file name; using last part after -")
        good_name = fragments[-1]
    # Make sure that this fragment ends with .pkl;
    # otherwise we made a false split, there was a - in the desired name
    # So add all the remaining parts until we hit the pkl name or the end
    else:
        while not good_name.endswith(".pkl") and idx < len(fragments) - 1:
            if fragments[idx + 1] not in ["modified.pkl", "final.pkl"]:
                good_name += fragments[idx + 1]
            else:
                good_name += ".pkl"
            idx += 1

    # In case there is still no .pkl in the name (other file type)
    try:
        where_to_cut = good_name.index(".")
    except:
        pass
    else:
        good_name = good_name[:where_to_cut]

    return good_name


def find_peptide_concentration_names(index):
    """ Find the level names closest to Peptide and Concentration in the index.
    Warning: some level names are hardcoded here. The list will need to be
    updated if new types of data come in.

    Args:
        (pd.MultiIndex): the index to look in.

    Returns:
        to_remove (list): level name closest to Peptide and level name
            closest to Concentration. Always has length 2.
    """
    # Potential level names equivalent to Peptide and Concentration,
    # in decreasing order of priority
    peptide_lvl_names = ["Peptide", "TumorPeptide"]
    concentration_lvl_names = ["Concentration", 'IFNgPulseConcentration',
        'TumorCellNumber', 'TCellNumber']
    existing = index.names
    to_remove = []

    for nm in peptide_lvl_names:
        if nm in existing:
            to_remove.append(nm)
            break  # don't look for the less important ones.
    if len(to_remove) == 0:
            to_remove.append("")
    for nm in concentration_lvl_names:
        if nm in existing:
            to_remove.append(nm)
            break
    if len(to_remove) == 1:
        to_remove.append("")

    return to_remove


def index_split(mindex, to_ignore):
    """ Return a list of non-empty labels in the index with the levels
    in to_ignore removed.

    Args:
        mindex (pd.MultiIndex): the DataFrame whose index will be reduced partially.
        to_ignore (list): list of level names to remove.

    Returns:
        index_entries (list): list of labels without the ignored levels.
        not_removed (list): list of levels that could not be found in the index
    """
    level_names = mindex.names
    not_removed = []
    remove_everything = False  # whether something will be left in the index
    for lvl in to_ignore:
        try:
            mindex = mindex.droplevel(lvl)
        except KeyError as e:
            not_removed.append(lvl)
        except ValueError:
            # If all the entries in the index are in to_remove,
            # the last one cannot be removed
            if len(mindex.names) == 1 and mindex.names[0] == lvl:
                remove_everything = True

    if not remove_everything:
        index_entries = list(mindex.unique())
        # If there was only one extra label level, turn manually each
        # label into a list of length one.
        if type(index_entries[0]) != tuple:
            index_entries = [(a,) for a in index_entries]
    else:
        index_entries = []

    return index_entries, not_removed


### Processing function copied and modified here to return df of spline objects
def process_file(folder, file, **kwargs):
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
        spline_frame (pd.DataFrame): spline objects
        df (pd.DataFrame): concentrations, integrals, etc.
    """
    # Processing-related keyword arguments
    take_log = kwargs.get("take_log", True)
    rescale_max = kwargs.get("rescale_max", False)
    smooth_size = kwargs.get("smooth_size", 3)
    rtol_splines = kwargs.get("rtol_splines", 1/2)
    max_time = kwargs.get("max_time", 72)

    # Import raw data
    data = pd.read_hdf(os.path.join(folder, file))

    # Put all timepoints for a given cytokine in continuous columns
    data = data.stack().unstack('Cytokine')

    # Check for randomly or structurally missing datapoints and interpolate between them
    data = treat_missing_data(data)

    # Import the limits of detection, if any
    cytokine_lower_lod = lod_import(file[32:40])

    # Take the log of the data if take_log, else normalize in linear scale
    data_log = log_management(data, take=take_log, rescale=rescale_max, lod=cytokine_lower_lod)

    # Smooth the data points before fitting splines for interpolation
    data_smooth = smoothing_data(data_log, kernelsize=smooth_size)

    # Fit cubic splines on the smoothed series
    spline_frame = generate_splines(data_log, data_smooth,rtol=rtol_splines)

    # Return data in various stages of processing
    return [data, data_log, data_smooth, spline_frame]
