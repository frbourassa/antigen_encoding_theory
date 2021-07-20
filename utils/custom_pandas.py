"""
Small module with functions extending or making more convenient the slicing
capabilities of pandas, which I often find not meeting my needs out of the box.

@author:frbourassa
November 5, 2020
"""
import pandas as pd

def xs_slice(df, name, lvl_slice, axis=0):
    """ Similar to the pandas xs method, but to select a slice 'lvl_slice' on the level 'name'
    instead of a single entry. It does not reduce the number of levels like xs.
    The main requirement is that the bounds specified in the slicer be in the index.
    This is intuitive for string labels, but not for numerical ones (we would like
    to include any value within the bounds; but pandas index don't work like this. )
    Be sure to choose as slice bounds the first and last elements satisfying
    the inequality you want to implement.

    lvl_slice can also be a list (or array) of labels to keep in the level name,
    since pandas also accepts multi-level indexes made up of a mix of slice
    objects and lists of labels.
    See https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#using-slicers

    Args:
        df (pd.DataFrame)
        name (str or int): the name on the level on which to slice
        lvl_slice (slice or list): a slice object to apply on the selected
            level, or a list of labels to keep in that level.
        axis (int): 0 for slicing the index, 1 for slicing the columns.

    Returns:
        df_s (pd.DataFrame): the df, sliced, still with all the original index levels.

    Note this could also be done with a (clumsy) one-liner:
        df[df.index.get_level_values('name').isin(lvl_slice)]
    because the df.index.get_level_values('name').isin(lvl_slice)
    creates a list of bool of length == number of rows, with True
    only in rows where the label in level 'name' is in lvl_slice.

    Or even better, with the syntax (also works with .loc):
        df.iloc[df.index.isin(['stock1','stock3'], level=name)]
    Or even just (because df[l] if l is a list slices on the index,
                even if df[l] with l a label slices on columns):
        df[df.index.isin(['stock1','stock3'], level=name)]
    See answer to question 2b and second reply in:
        https://stackoverflow.com/questions/53927460/select-rows-in-pandas-multiindex-dataframe

    But I prefer having this xs function, which is ~2x faster! Sample run
    on a df with 198144 rows

    In [3]: df
    Out[3]:
                                                                                                         Node 1    Node 2
    Data                         TCellNumber Peptide Concentration Time Processing type Feature
    Activation_2                 100k        A2      100nM         1    Fit             concentration  0.001796 -0.001099
                                                                                        integral       0.001796 -0.001099
                                                                        Splines         concentration -0.002662 -0.005222
                                                                                        integral      -0.002662 -0.005222
                                                                   2    Fit             concentration  0.005264 -0.002983
    ...                                                                                                     ...       ...
    TCellNumber_OT1_Timeseries_7 3k          V4      1uM           71   Splines         integral      -0.637188 -1.451083
                                                                   72   Fit             concentration -0.006810 -0.026964
                                                                                        integral      -0.332882 -1.373057
                                                                        Splines         concentration -0.011938 -0.035262
                                                                                        integral      -0.649126 -1.486345

    [198144 rows x 2 columns]


    In [4]: %timeit df[df.index.isin(["N4", "A2"], level="Peptide")]
    6.41 ms ± 63.1 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

    In [5]: %timeit df.iloc[df.index.isin(["N4", "A2"], level="Peptide")]
    6.26 ms ± 18.8 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

    In [6]: %timeit df.loc[df.index.isin(["N4", "A2"], level="Peptide")]
    6.21 ms ± 42 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

    In [7]: import custom_pandas as custom_pd

    In [8]: %timeit custom_pd.xs_slice(df, name="Peptide", lvl_slice=["N4", "A2"], a
       ...: xis=0)
    2.92 ms ± 95.5 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

    In [9]: %timeit df[df.index.get_level_values("Peptide").isin(["N4", "A2"])]
    6.59 ms ± 70.6 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    """
    # Find the position of the specified level
    idx = df.index if axis == 0 else df.columns
    try:
        spec_level = idx.names.index(name)
    except ValueError:
        raise KeyError("{} is not in the index {}".format(name, idx))

    # Build a slicer with the specified lvl_slice at the right position.
    i = 0
    slicer = []
    while i < spec_level:
        slicer.append(slice(None))
        i += 1
    slicer.append(lvl_slice)
    if len(slicer) > 1:
        slicer = tuple(slicer)
    else:
        slicer = slicer[0]

    # Slice the dataframe on the required axis
    df_s = df.loc(axis=axis)[slicer]
    return df_s
