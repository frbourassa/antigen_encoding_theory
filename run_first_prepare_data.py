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

def clean_up_split_files(bak_file, original_files=[],
                                folder=os.path.join("data", "initial/")):
    """ Convert any new .bak file back to its original state (.hdf)
    and delete files that were created from its splitting, if they
    were not in the original repository's list of files. That list is
    optional; if not provided, all bak files are put back in place
    and their subfiles are deleted.
    """
    if not bak_file.endswith(".bak"):
        raise ValueError("{} is not a .bak file.".format(bak_file))
    # Find the file's nice name.
    # Assuming all names start with cytokine...-date-...
    nicename_bak = "-".join(bak_file.split("-")[2:])
    nicename_bak = nicename_bak[:-4]  # Remove .bak

    # Find subfiles with the same nice name
    # If they were not in the original_repo, delete them
    new_files = os.listdir(folder)
    for f in new_files:
        # Check if the nicename is just nicename_bak with -index.
        nicename = "-".join(f.split("-")[2:-1])
        # if so, we have a subfile, to delete if it did not exist before
        if (nicename == nicename_bak):
            if f not in original_files:
                os.remove(os.path.join(folder, f))
            else:
                print("Found replicate {}".format(f, bak_file)
                + " but did not delete because it was already there")

    # If the bak file itself was not already there, revert it back
    # We know its name ends in .bak, so we can slice the string to remove .bak
    if bak_file not in original_files:
        # Find the original in the original list
        # look for same name except extension by including the . in startswith
        orig_file = [a for a in original_files if a.startswith(bak_file[:-3])]
        try:
            orig_file = orig_file[0]
        # Apparently that file did not exist in another state.
        except:
            print("Original file did not exist, leaving bak as it is")
        # Can revert to old extension
        else:
            old_ext = orig_file.split(".")[-1]
            os.rename(os.path.join(folder, bak_file),
                os.path.join(folder, bak_file[:-4] + "." + old_ext))
            print("Converted file {} back to {} format".format(bak_file, old_ext))
    else:
        print("file {} was kept as .bak because it existed".format(bak_file))
    return 0

if __name__ == "__main__":
    # Set to False if there are only a few new datasets to process
    # True by default: all dataframes are processed and written.
    overwrite = True
    # Adapt dataframes in data/initial/
    initial_folder = os.path.join("data", "initial/")
    original_contents = os.listdir(initial_folder)

    print("Starting to adapt dataframes...\n")
    main_adapt(initial_folder)
    print("\nFinished adapting initial/ dataframes into final/")
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


    # After saving the final/ files, remove split files.
    print("\nDone saving to data/processed/")
    print("Cleaning up split files to leave data in initial/ unchanged...\n")
    for bkf in os.listdir(os.path.join("data/", "initial/")):
        if bkf.endswith(".bak"):
            clean_up_split_files(bkf, original_files=original_contents,
                            folder=os.path.join("data", "initial/"))

    print("Done!")
