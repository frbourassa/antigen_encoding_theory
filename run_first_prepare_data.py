""" First script to run to reshape and reindex the raw cytokine concentration
data available as HDF5 files in data/initial/.
The formatted dataframes are then saved in data/final/, also in HDF5 format;
they are used for a few theory applications in this repository.

This script then preprocesses the raw cytokine time series with the steps
described in supplementary information: log-transformation,
smoothing interpolation, and time integration.

Run this script from the top folder in the antigen_encoding_theory repo, i.e.
from the folder antigen_theory_encoding/, which contains the present script.

@author:frbourassa
November 2021
"""
import numpy as np
import pandas as pd
import os

from ltspcyt.scripts.adapt_dataframes import main_adapt
from ltspcyt.scripts.process_raw_data import process_file

if __name__ == "__main__":
    # Set to False if there are only a few new datasets to process
    # True by default: all dataframes are processed and written.
    overwrite = False
    # Adapt dataframes in data/initial/
    initial_folder = os.path.join("data", "initial/")

    print("Starting to adapt dataframes...")
    main_adapt(initial_folder)
    print("Finished adapting initial/ dataframes into final/")
    print("Starting log-smoothing-integral processing...")

    # One can modify processing arguments here. The recommended values
    # are the default ones, reproduced here:
    process_kwargs = {
        "take_log": True,
        "rescale_max": False,
        "smooth_size": 3,
        "rtol_splines": 0.5,
        "max_time": 72
    }

    # Process adapted dataframes. This may take a minute to complete.
    final_folder = os.path.join("data", "final/")
    for f in os.listdir(final_folder):
        if f.endswith(".hdf") or f.endswith(".h5"):
            # Extract the short file name from the full name, for saving
            # Usual format is cyto...-date-filename-final.hdf
            # Maybe there are -- in the filename, e.g. HighMI_1-i,
            # so allow a range and join if necessary
            nicename = "-".join(f.split("-")[2:-1])
            filename = os.path.join("data", "processed", nicename + ".hdf")
            # Check if it already exists, skip processing if overwrite==False
            if not os.path.isfile(filename) or overwrite:
                res = process_file(final_folder, f, **process_kwargs)
                [data, data_log, data_smooth, df_features] = res
                # Save processed file
                df_features.to_hdf(filename, key="df")
    print("Done!")
