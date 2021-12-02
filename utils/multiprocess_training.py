import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from utils.process_raw_data import process_file_choices, select_naive_data

# Slicing utility
def slice_level(df, special_slice, target_lvl="Time", axis=0):
    allslices = []
    idx = df.index if axis == 0 else df.columns
    for lvl in idx.names:
        sl = special_slice if lvl == target_lvl else slice(None, None)
        allslices.append(sl)
    if axis == 0:
        return df.loc[tuple(allslices),]
    elif axis ==1:
        return df.loc[:, tuple(allslices)]

# Global variables common to the two functions below
# Put them in one initialization function called by each other function
# to get a consistent set of peptides, concentrations and cytokines.
def init_peps_cytos_concs():
    # Keep the train_peptides order as in the original code. E1, weakest, is first.
    train_peptides = ["N4", "Q4", "T4", "V4", "G4", "E1"][::-1]
    keep_cytokines = ["IFNg", "IL-17A", "IL-2", "IL-6", "TNFa"]
    keep_conc = ["1uM", "100nM", "10nM", "1nM"]
    keep_cytokines.sort()
    return train_peptides, keep_cytokines, keep_conc

def train_classifier(data, hidden_sizes=(2,), seed=None, activ="tanh"):
    train_peptides, keep_cytokines, keep_conc = init_peps_cytos_concs()
    peptide_dict = {k:v for v, k in enumerate(train_peptides)
                         if k in data.index.get_level_values("Peptide").unique()}

    #Extract times and set classes
    y = data.index.get_level_values("Peptide").map(peptide_dict)

    mlp = MLPClassifier(activation=activ, hidden_layer_sizes=hidden_sizes, max_iter=5000,
            solver="adam", random_state=seed, learning_rate="adaptive", alpha=0.01).fit(data, y)

    score = mlp.score(data, y)
    return mlp, score


def test_classifier(mlp, data):
    train_peptides, keep_cytokines, keep_conc = init_peps_cytos_concs()
    peptide_dict = {k:v for v, k in enumerate(train_peptides)
                         if k in data.index.get_level_values("Peptide").unique()}

    #Extract times and set classes
    y = data.index.get_level_values("Peptide").map(peptide_dict)

    score = mlp.score(data, y)
    return score

def crossvalidate_classifier(data, hidden_sizes=(2,), seed=None, activ="tanh"):
    train_peptides, keep_cytokines, keep_conc = init_peps_cytos_concs()
    peptide_dict = {k:v for v, k in enumerate(train_peptides)
                         if k in data.index.get_level_values("Peptide").unique()}

    #Extract times and set classes
    y = data.index.get_level_values("Peptide").map(peptide_dict)

    mlp = MLPClassifier(activation=activ, hidden_layer_sizes=hidden_sizes, max_iter=5000,
            solver="adam", random_state=seed, learning_rate="adaptive", alpha=0.01)

    scores = cross_validate(mlp, data.values, y, cv=5, return_train_score=True, return_estimator=False)
    return scores


def process_train_dsets(file_list, process_args, folder="data/",
    tslice=slice(1, 71), extra_cytos=[]):
    train_peptides, keep_cytokines, keep_concs = init_peps_cytos_concs()
    for c in extra_cytos:
        if c not in keep_cytokines:
            keep_cytokines.append(c)
    all_dfs = {}
    for f in file_list:
        df = process_file_choices(folder, f, **process_args)
        # Keep naive data only (like import_WT_output function)
        df = select_naive_data(df)
        # Keep relevant cytokines only
        df = df.loc[:, df.columns.isin(keep_cytokines, level="Cytokine")]
        # Keep training peptides only
        df = df.loc[df.index.isin(train_peptides[::-1], level="Peptide")]
        # Keep 100k T cells only, which was done manually in the original network's training
        df = df.xs("100k", level="TCellNumber", drop_level=False)
        # Keep typical concentrations
        df = df.loc[df.index.isin(keep_concs, level="Concentration")]
        # Append dataframe
        short_name = f[41:-10]  # Names are formatted uniformly
        all_dfs[short_name] = df
    full_df = pd.concat(all_dfs, names=["Data"], axis=0)
    # Keep only the training times before computing the min and max for normalization
    full_df = slice_level(full_df, tslice, target_lvl="Time")
    # Normalize
    dfmin, dfmax = full_df.min(axis=0), full_df.max(axis=0)
    full_df = (full_df - dfmin) / (dfmax - dfmin)
    dfminmax = pd.concat({"min": dfmin, "max":dfmax}, names=["Extremum"], axis=1)
    # Return the normalization factors too, to be able to reverse the scaling afterwards
    return full_df, dfminmax

def process_test_dsets(file_list, process_args, dfminmax, folder="data/",
    tslice=slice(1, 71), extra_cytos=[]):
    train_peptides, keep_cytokines, keep_concs = init_peps_cytos_concs()
    for c in extra_cytos:
        if c not in keep_cytokines:
            keep_cytokines.append(c)
    all_dfs = {}
    for f in file_list:
        df = process_file_choices(folder, f, **process_args)
        # Keep naive data only (like import_WT_output function)
        df = select_naive_data(df)
        # Keep relevant cytokines only
        df = df.loc[:, df.columns.isin(keep_cytokines, level="Cytokine")]
        # Keep training peptides only
        df = df.loc[df.index.isin(train_peptides[::-1], level="Peptide")]
        # Keep 100k T cells only, which was done manually in the original network's training
        df = df.xs("100k", level="TCellNumber", drop_level=False)
        # Keep typical concentrations
        df = df.loc[df.index.isin(keep_concs, level="Concentration")]
        # Append dataframe
        short_name = f[41:-10]  # Names are formatted uniformly
        all_dfs[short_name] = df

    full_df = pd.concat(all_dfs, names=["Data"], axis=0)
    # Keep only the training times
    full_df = slice_level(full_df, tslice, target_lvl="Time")
    # Normalize with the training min and max
    dfmin, dfmax = dfminmax["min"], dfminmax["max"]
    full_df = (full_df - dfmin) / (dfmax - dfmin)
    return full_df

def process_train(lis, train_files, folder="data/"):  # l: log, s: smooth, i: integral
    process_kwargs = {"take_log": True, "do_smooth": True, "do_integrate": True,
                    "rescale_max": False, "max_time": 72, "smooth_size": 3, "rtol_splines": 1/2}
    train_time_slice = slice(1, 71)  # Process with 72 hours but exclude from training to mimic original
    # Process training data according to lis
    process_kwargs["take_log"] = lis[0]
    process_kwargs["do_smooth"] = lis[2]
    process_kwargs["do_integrate"] = lis[1]
    df_train, df_minmax = process_train_dsets(train_files, process_kwargs,
                                              folder=folder, tslice=train_time_slice)
    print("Done processing {}".format(lis))
    # Troubleshooting
    #print(df_train.index.names)
    #for nm in df_train.index.names:
        #print(df_train.index.get_level_values(nm).unique())

    # Now train a classifier with all the train data; this will
    # reproduce the original training in the case of log+smooth+integral
    sd = 90 + (1-int(lis[0]))*1 + (1-int(lis[1]))*2 + (1-int(lis[2]))*4
    classif, score = train_classifier(df_train, hidden_sizes=(2,), seed=sd, activ="tanh")

    # Also cross-validate to get a more robust estimate of the training score
    more_scores = crossvalidate_classifier(df_train, hidden_sizes=(2,), seed=sd+1329, activ="tanh")
    more_scores["whole_score"] = score

    # Print results
    print(lis)
    print("Number of points:", df_train.shape)  # For interpolated data, should be (5680, 5)
    print("Training score: {}".format(100*score))
    #print("Average training score:", np.mean(more_scores["train_score"])*100, "pm", np.std(more_scores["train_score"])*100)
    print()

    # Return the classifier, data, and normalization factors
    #all_classifiers[tuple(lis)] = classif
    #all_train_scores[tuple(lis)] = np.mean(more_scores)  # score
    #all_train_dfs[tuple(lis)] = df_train
    #all_minmax_dfs[tuple(lis)] = df_minmax
    return classif, more_scores, df_train, df_minmax
