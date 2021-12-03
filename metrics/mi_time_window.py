import numpy as np
import pandas as pd
import utils.custom_pandas as cpd

from metrics.discrete_continuous_info import discrete_continuous_info_fast, discrete_continuous_info_ref

def compute_mi_slice(df, q, t, window, knn=3, speed="fast"):
    """ Compute the mutual information between the variable X in the columns of df
    and the variable Q, a level in the index, in a time window centered at time t. 
    Time should be in the index too. Computes the MI between each feature and 
    the labels in q separately (sklearn's implementation does not handle 
    joint probability distributions). 
    
    Args:
        df (pd.DataFrame): should have "Time" in its column levels 
            and q in its index. 
        q (str): the name of the index level to use as labels specifying
            the values of the discrete RV. 
        t (float): the time on which the time window is centered. 
        window (float): duration of the time interval over which to 
            keep the samples. If a single time point is desired, use 0. 
        speed (str): either "fast" or "slow" ("slow" is ref. code)
    
    Returns:
        mi (float): mutual information
    """
    # Define the time window and slicer. Assuming the Time index is sorted. 
    tpts = np.array(df.index.get_level_values("Time").unique())
    tlo = tpts[np.searchsorted(tpts, t - window/2, side="left")]
    try:  # searchsorted can return the index next to the last; in that case, take the last
        thi = tpts[np.searchsorted(tpts, t + window/2, side="right")]
    except IndexError:
        thi = tpts[-1]
    time_slice = slice(tlo, thi)
    
    # Slice the df with our custom function
    df_t = cpd.xs_slice(df, "Time", time_slice, axis=0)
    if isinstance(df_t, pd.Series):
        df_t = df_t.to_frame()

    # Extract the labels about which the data should inform
    try:
        mapping = {a:i for i, a in enumerate(df_t.index.get_level_values(q).unique())}
        target = np.asarray(df_t.index.get_level_values(q).map(mapping))
    except ValueError:
        print("{} not in {}".format(q, df_t.index))
    
    # Compute the MI!
    if speed == "fast":
        mi = discrete_continuous_info_fast(target, df_t.values, k=knn, base=2)
    elif speed == "slow":
        mi, _ = discrete_continuous_info(target, df_t.values, k=knn, base=2)
    else:
        raise ValueError("Speed '{}' not available".format(speed))
        
    return mi


# Compute the maximum possible MI
# Function taken from the code for phi-evo simulations with cytokines
# Can input the Peptide level values to this, it counts the number of 
# occurences of each label
def entropy(xvals):
    """ Builds the distribution and compute the entropy
    of the sample xvals of some random variable X
    Args:
        xvals (list of int): list of sampled values of the X variable
    Returns:
        info (float): the entropy of the distribution from the sample, in bits
    """
    possible_values = list(set(xvals))
    mapping = {possible_values[i]:i for i in range(len(possible_values))}
    probs = np.zeros(len(possible_values))
    for x in xvals:
        probs[mapping[x]] += 1
    probs = probs / np.sum(probs)
    # We are sure that no prob value is zero because we only considered possible values
    info = np.sum(-probs * np.log(probs)) / np.log(2)  # in bits
    return info

def find_nearest(my_array, target, condition):
    """ Nice function by Akavall on StackOverflow:
    https://stackoverflow.com/questions/17118350/how-to-find-nearest-value-that-is-greater-in-numpy-array
    Page consulted Feb 10, 2019.
    """
    diff = my_array - target
    if condition == "above":
        # We need to mask the negative differences and zero
        # since we are looking for values above
        mask = (diff <= 0)
    elif condition == "below":
        # We need to mask positive differences and zero
        mask = (diff >= 0)
    if np.all(mask):
        return None # returns None if target is greater than any value
    masked_diff = np.ma.masked_array(diff, mask)
    return masked_diff.argmin()

def compute_mi_timecourse(dfs, q, overlap, window, knn=3, names=None, speed="fast"):
    """ Compute the MI over time for each feature in each df. 
    
    Args:
        dfs (dict of pd.DataFrames): each df has one feature per column, and q in its index. 
        q (str): name of the index level to use as the discrete RV
        overlap (bool): whether to allow the use of overlapping time windows
            when computing the MI at succesive time points. 
        window (float): the time duration over which to take the samples to compute the
            MI at one time point. Set to 0 if the distribution is to be based on single time points. 
        knn (int): number of nearest-neighbors to use. 
        speed (str): "fast" or "slow" (where "slow" is the reference code)
        
    Returns:
        
    """      
    # Set an order in which the dfs will be processed
    names = list(dfs.keys())
    
    # Determine the upper bound on the MI
    maximum_mi = entropy(dfs[names[0]].index.get_level_values("Peptide"))

    # Determine the times to consider for each df
    time_values = [dfs[a].index.get_level_values("Time").unique() for a in names]
    # Define the central times to consider
    if overlap or window == 0:  # we allow all times
        times = time_values
    else:  # we want different times at each evaluation
        times = []
        for t_array in time_values:
            tstart = t_array[find_nearest(t_array, t_array[0] + window/2, "above")]
            times.append(np.arange(tstart, t_array[-1], window))
    
    # Compute the MI at each selected time point, for each df
    mi_courses = []  # Contains the MI time course of each df variable
    for i in range(len(names)):
        # Will be an array with one time per row, one feature per column
        current_course = []
        df = dfs[names[i]]
        for t in times[i]:
            current_course.append(compute_mi_slice(df, q, t=t, window=window, knn=knn, speed=speed))
            if speed == "slow":
                print("Time {} h done".format(t))
        mi_courses.append(np.array(current_course))
    
    df_mi_courses = pd.concat([pd.Series(mi_courses[i], index=times[i]) for i in range(len(times))], 
                             keys=names, names=["Variable", "Time"])
    print(df_mi_courses.index.names)
    df_mi_courses = df_mi_courses.unstack("Variable")
    
    return df_mi_courses, maximum_mi