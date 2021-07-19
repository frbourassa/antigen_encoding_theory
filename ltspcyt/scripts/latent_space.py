#! /usr/bin/env python3
"""Project data in latent space of neural network"""

import warnings
warnings.filterwarnings("ignore")

import os
import pickle,sys
import numpy as np
import pandas as pd
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tkinter as tk
import seaborn as sns
from ltspcyt.scripts.adapt_dataframes import set_standard_order


path = ""
def import_mutant_output(mutant, folder=path+"data/processed/"):
    """Import processed cytokine data from experiments that contains
        "mutant" conditions.

    Args:
        mutant (str): name of file with mutant data.
                Has to be one of the following: "Tumor", "Activation",
                    "TCellNumber", "Macrophages", "CAR", "TCellType",
                    "CD25Mutant", "ITAMDeficient", "Drug"
        folder (str): full path to folder containing .hdf files to parse

    Returns:
        df_full (dataframe): the dataframe with processed cytokine data
    """

    naive_level_values={
        "ActivationType": "Naive",
        "Antibody": "None",
        "APC": "B6",
        "APCType": "Splenocyte",
        "CARConstruct": "None",
        "CAR_Antigen": "None",
        "Genotype": "WT",
        "IFNgPulseConcentration": "None",
        "TCellType": "OT1",
        "TLR_Agonist": "None",
        "TumorCellNumber": "0k",
        "DrugAdditionTime": 36,
        "Drug": "Null",
        "ConditionType": "Control",
        "TCR": "OT1"
    }

    mutant_levels={
        "Tumor": ["APC", "APCType", "IFNgPulseConcentration"],
        "Activation": ["ActivationType"],
        "TCellNumber": [],
        "Macrophages": ["TLR_Agonist", "APCType"],
        "CAR":["CAR_Antigen", "Genotype", "CARConstruct"],
        "TCellType": ["TCellType", "TCR"],
        "CD25Mutant": ["Genotype"],
        "ITAMDeficient": ["Genotype"],
        "NewPeptide": [],
        "Drug": ["Drug", "DrugAdditionTime"],
        "hTCR":['Donor']
    }

    essential_levels=["TCellNumber", "Peptide", "Concentration", "Time"]

    dfs_dict = {}
    for file in os.listdir(folder):
        if (mutant not in file) | (not file.endswith(".hdf")):
            continue
        df = pd.read_hdf(folder + file)

        # If level not in essential levels or mutant-required level,
        # keep naive level values and drop level
        for level in df.index.names:
            if level not in essential_levels + mutant_levels[mutant]:
                df = df[df.index.get_level_values(level) == naive_level_values[level]]
                df = df.droplevel(level, axis=0)
        dfs_dict[file[:-4]] = df
        print(file)
        print(df.index.names)

    # Concatenate all saved dataframes
    df_full = pd.concat(dfs_dict, names=["Data"])
    return df_full
