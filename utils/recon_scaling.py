""" Function used to rescale properly reconstructed cytokine data. Includes
functions to import raw data without much processing, to extract the absolute
lower limit of detection and put back absolute cytokine concentrations.

@author: frbourassa
2021
"""
import pandas as pd
import numpy as np
import os

# Extract naive OT-1 data only to estimate noise in cytokine data
def extract_process_naive_part(df, cytokines=[]):
    """
    Args:
        df (pd.DataFrame): raw, but formatted, data to process
    Returns:
        df_log (pd.DataFrame): naive OT-1 data, log10 and rescaled
        min_concs (pd.Series): min. conc. of cytokines by which we rescaled
    """
    if cytokines == []:
        cytokines = ["IFNg", "IL-17A", "IL-2", "IL-6", "TNFa"]
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

    # Remove all unwanted (non-naive OT1) index level and labels
    mask=[True] * len(df)
    for index_name in df.index.names:
        if index_name in naive_pairs.keys():
            mask=np.array(mask) & np.array([index == naive_pairs[index_name]
                        for index in df.index.get_level_values(index_name)])
            df=df.droplevel([index_name])
    df = df[mask]

    # Remove all unwanted cytokines
    df = df.loc[df.index.isin(cytokines, level="Cytokine")].unstack("Cytokine").stack("Time")

    # Rescale by the minimum concentration (lower LOD) and take the log10
    min_concs = df.min(axis=0)
    df_log = np.log10(df / min_concs)

    return df_log, min_concs


def import_folder_naive_data(folder, datalist):
    fileNameDict = {}
    fileDateDict = {}
    # First filter the list of file names
    for fileName in os.listdir(folder):
        if fileName.endswith("-final.pkl") and fileName in datalist:
            fileNameDict[fileName[41:-10]] = fileName
            fileDateDict[fileName[41:-10]] = fileName.split("-")[1]
    sortedFileNames = sorted(fileNameDict.keys(), key=fileDateDict.get)

    # Then, import the naive data, drop everything unwanted, concatenate.
    df_full = {}
    df_min_concs = {}
    for fileName in sortedFileNames:
        fullFileName = fileNameDict[fileName]
        try:
            df = pd.read_pickle(os.path.join(folder, fullFileName))
        except:
            print("Could not load", fileName)
            continue
        df, min_concs = extract_process_naive_part(df)
        df_full[fileName] = df
        df_min_concs[fileName] = min_concs
    df_full = pd.concat(df_full, names=["Data"])
    df_min_concs = pd.concat(df_min_concs, names=["Data"])
    print("Loaded all available datasets")
    return df_full, df_min_concs


def scale_back(df_cyt, dfmin, dfmax):
    """ Take scaled cytokine data/reconstruction, and put it back in
    log_10(pM) scale.
    """
    feat_keys = ["integral", "concentration", "derivative"]
    df_scaled = df_cyt.copy()
    for typ in feat_keys:
        try:
            df_scaled[typ] = df_cyt[typ] * (dfmax - dfmin)
            if typ == "integral":
                df_scaled[typ] += dfmin
        except KeyError:
            continue  # This feature isn't available in this df; fine
        else:
            print("Put scale back for feature", typ)
    return df_scaled
