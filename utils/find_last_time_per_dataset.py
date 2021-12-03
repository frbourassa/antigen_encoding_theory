""" A simple script that opens each raw experimental dataframe in a folder,
finds the last time point available for all conditions in that experiment, and
saves it to a summary json file in fit_results/

@author: frbourassa
November 2021
"""
import os
import numpy as np
import pandas as pd
import json

def extract_nice_name(s):
    if s.startswith("cytokineConcentrationPickleFile"):
        i1 = 41
    else:
        i1 = 0
    if s.endswith("final.hdf"):
        i2 = -10
    elif s.endswith("modified.hdf"):
        i2 = -13
    elif s.endswith(".hdf"):
        i2 = -4  # keep everything except .pkl
    return s[i1:i2]

def find_last_times(folder=os.path.join("data", "final"),
                    where_save=None):
    """ Short script parsing a folder of hdf data files, to build a dictionary
    of the last time point available in each data set.
    folder is the folder to parse
    where_save is the path and name of the file where to save the
    dictionary as a JSON file, optionally.
    """
    max_times_ser = {}
    for f in os.listdir(folder):
        if not f.endswith(".hdf"): continue
        df = pd.read_hdf(os.path.join(folder, f), key="df")
        try:
            assert df.columns.name == "Time"
            # Keep only times which have data for every condition.
            df = df.dropna(axis=0)
            times = df.columns.get_level_values("Time").unique().astype(float)
        except AssertionError:
            print("Columns are not Time in {}".format(f))
        except KeyError:
            print("Could not find Time in columns of {}".format(f))
        else:
            last_time = max(times)
            max_times_ser[extract_nice_name(f)] = last_time
        print("Done file {}".format(f))

    if where_save is not None:
        with open(where_save, "w") as h:
            json.dump(max_times_ser, h)
    return max_times_ser


if __name__ == "__main__":
    find_last_times(folder=os.path.join("data", "final"),
                    where_save=os.path.join("data", "misc",
                        "last_time_point_per_experiment.json"))
