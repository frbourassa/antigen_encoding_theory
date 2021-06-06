#! /usr/bin/env python3
"""Train and save a neural network"""

import os
import pickle
import sys
import numpy as np
import pandas as pd
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
from sklearn import neural_network
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from ltspcyt.scripts.adapt_dataframes import set_standard_order


path = ""
idx = pd.IndexSlice

def import_WT_output(folder=path+"data/processed/"):
    """Import splines from wildtype naive OT-1 T cells by looping through all datasets

    Returns:
            df_full (dataframe): the dataframe with processed cytokine data
    """

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

    dfs_dict = {}
    for file in os.listdir(folder):
        if not file.endswith(".hdf"):
            continue

        df=pd.read_hdf(folder + file)
        mask=[True] * len(df)

        for index_name in df.index.names:
            if index_name in naive_pairs.keys():
                mask=np.array(mask) & np.array([index == naive_pairs[index_name] for index in df.index.get_level_values(index_name)])
                df=df.droplevel([index_name])
        dfs_dict[file[:-4]] = df[mask]
        print(file)
    # Concatenate all dfs
    df_full = pd.concat(dfs_dict, names=["Data"])
    return df_full

def plot_weights(mlp,cytokines,peptides,**kwargs):

    fig,ax=plt.subplots(1,2,figsize=(8,2))
    ax[0].plot(mlp.coefs_[0],marker="o",lw=2,ms=7)
    ax[1].plot(mlp.coefs_[1].T[::-1],marker="o",lw=2,ms=7)
    [a.legend(["Node 1","Node 2"]) for a in ax]

    ax[0].set(xlabel="Cytokine",xticks=np.arange(len(cytokines)),xticklabels=cytokines,ylabel="Weights")
    ax[1].set(xlabel="Peptide",xticks=np.arange(len(mlp.coefs_[1].T)),xticklabels=peptides)
    plt.savefig("%s/weights.pdf"%tuple(kwargs.values()),bbox_inches="tight")
