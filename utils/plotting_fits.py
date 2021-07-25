import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerBase
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
import matplotlib.text as mtext
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb
from colorsys import rgb_to_hls


peptide_ranks = {"N4":13, "Q4":12, "T4":11, "V4":10, "G4":9, "E1":8,
              "A2":7, "Y3":6, "A8":5, "Q7":4}

dico_units = {
    'f':1e-15,   # femto
    "p": 1e-12,  # pico
    'n':1e-9,    # nane
    'u':1e-6,    # micro
    'm':1e-3,    # milli
    'c':1e-2,    # centi
    'd':0.1,     # deci
    '':1.,
}
def read_conc(text):
    """ Convert a concentration value in text into a float (M units). """
    # Separate numbers from letters
    value, units = [], []
    for a in text:
        if a.isnumeric():
            value.append(a)
        elif a.isalpha():
            units.append(a)

    # Put letters together and numbers together
    value = ''.join(value)
    units = ''.join(units)
    if value is not '':
        conc = float(value)
    else:
        conc = 0  # text is 'Blank' or 'n/a'

    # Read the order of magnitude
    units = units.replace("M", '')

    # If we encounter a problem here, put a value that is impossibly large
    if len(units) == 1:
        conc *= dico_units.get(units, 1e6)
    elif conc > 0:  # conc == 0 if blank
        conc = 1e6

    return conc

# For adding subtitles in legends.
# Artist class: handle, containing a string
class LegendSubtitle(object):
    def __init__(self, message, **text_properties):
        self.text = message
        self.text_props = text_properties
        self.labelwidth = " "*(len(self.text)+3)
    def get_label(self, *args, **kwargs):
        return self.labelwidth  # no label, the artist itself is the text

# Handler class, give it text properties
class LegendSubtitleHandler(HandlerBase):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle.text, size=fontsize, **orig_handle.text_props)
        # Update the (empty) label to have a length enough to cover the whole artist text box
        # orig_handle.labelwidth = " "*int(np.ceil(len(orig_handle.text) + legend.handlelength))
        handlebox.add_artist(title)
        # Make the legend box wider if needed

        return title


### Color preparation functions

# A tweak around light and dark palettes
def create_cmap_seed(seedclr, n_colors=4, as_cmap=False, light=True):
    seedclr_hls = rgb_to_hls(r=seedclr[0], g=seedclr[1], b=seedclr[2])
    # Choose max (if light) or min (if dark) lightness
    start_lightness = min(seedclr_hls[1], 0.6) if light else max(seedclr_hls[1], 0.7)
    seedclr = sns.set_hls_values(seedclr, l=start_lightness, s=seedclr_hls[2], h=seedclr_hls[0])

    # Remove the last color, too dark
    if light:
        cmap = sns.light_palette(seedclr, n_colors=n_colors+1, as_cmap=as_cmap)[1:]
    else:
        cmap = sns.dark_palette(seedclr, n_colors=n_colors+1, as_cmap=as_cmap)[1:]
    return cmap

# Custom palette where the seed color is in the middle
def create_midseeded_clist(seedclr, n_colors=4, min_l=0.3, max_l=0.85):
    seedclr_hls = rgb_to_hls(r=seedclr[0], g=seedclr[1], b=seedclr[2])

    # Favour darker colors
    midpoint = n_colors // 2

    # lightnesses below the seed
    l = min(min_l, seedclr_hls[1])  # min l
    lincrem = (seedclr_hls[1] - l) / max(1, midpoint)
    clist = []
    while l < seedclr_hls[1]-0.01*lincrem:
        clist.append(sns.set_hls_values(seedclr, l=l))
        l += lincrem
    # Seed color in the middle
    clist.append(seedclr)
    assert abs(l - seedclr_hls[1]) < 1e-10
    l = seedclr_hls[1]

    # Lightnesses above the seed
    maxl = max(max_l, seedclr_hls[1])
    lincrem = (maxl - l) / max(1, (midpoint-1))
    l += lincrem
    while l < maxl + 0.01*lincrem:
        clist.append(sns.set_hls_values(seedclr, l=l))
        l += lincrem
    return clist


def add_hue_size_style_legend(ax, hues, sizes, styles, sizes_name, styles_name,
                                hue_sort_key=None, **kwargs):
    """ Add a legend in the given ax or fig containing the existing handles (expected
        to be solid lines) and sections explaining the types of curves.

        Tip: use kwargs to pass in small fontsize and handlelength,
            for Science figure format.
    Args:
        hues (dict): where keys are labels controlling hues, values are
            associated colors
        sizes (dict): where keys are labels controlling sizes, values are
            associated line widths
        styles (dict): where keys are labels controlling line styles,
            values are associated line styles.
        sizes_name (str): name of the feature controlling size
        styles_name (str): name of the feature controlling style
        kwargs: keyword arguments passed directyl to ax.legend, like
            handlelength and fontsize

    Returns:
        matplotlib.legend.Legend: the legend created on the ax.
    """
    # Hues: no subtitle, this is the kind of lines expected.
    # Make solid lines of width equal to the max in sizes
    handles = []
    if len(sizes) > 0:
        maxwidth = max(sizes.values())
    else:
        maxwidth = 2.

    if hue_sort_key is not None:
        sorted_hue_keys = sorted(hues.keys(), key=hue_sort_key, reverse=True)
    else:
        sorted_hue_keys = hues.keys()

    for k in sorted_hue_keys:
        handles.append(Line2D([0], [0], color=hues[k], label=k,
                              linestyle="-", linewidth=maxwidth))

    # Line sizes: it could be an empty dict if there is a single size and
    # we do not want it in the legend
    if len(sizes) > 0:
        handles.append(LegendSubtitle(sizes_name))
        sorted_keys = sorted(sizes.keys(), key=read_conc, reverse=True)
        for k in sorted_keys:
            handles.append(Line2D([0], [0], color="k", label=k,
                                  linestyle="-", linewidth=sizes[k]))

    # Line styles
    if len(styles) > 0:
        handles.append(LegendSubtitle(styles_name))
        sorted_keys = list(styles.keys())
        if "-" in sorted_keys:
            sorted_keys.remove("-")
            sorted_keys.insert(0, "-")
        for k in sorted_keys:
            handles.append(Line2D([0], [0], color="k", label=k,
                                 linestyle=styles[k], linewidth=maxwidth))

    leg = ax.legend(handles=handles,
              handler_map={LegendSubtitle: LegendSubtitleHandler()},
              loc="upper left", bbox_to_anchor=(0, 1), **kwargs)
    return leg


def create_labeling(df_feature, maxwidth=2.):
    """ Sort the index levels to get all the info needed for the legend.

    Returns:
        hue_info (list): lists of pairs of elements:
            [hue_level_name, hues],
            [size_level_name, sizes],
            [style_level_name, styles]
    """
    if isinstance(df_feature.index, pd.MultiIndex):
        index_names = df_feature.index.names
    else:
        index_names = [df_feature.index.name]
    print(index_names)

    # Check which levels are present
    for n in index_names:
        if n not in ["Peptide", "Concentration", "Data", "TCellNumber", 'Time', 'Processing type']:
            raise KeyError("I can't deal with level {}".format(n))
    if "Data" in index_names:
        if len(df_feature.get_level_values("Data").unique()) > 1:
            raise ValueError("Give a single data set!")

    # Hues per peptide
    default_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    try:
        peptide_order = sorted(df_feature.index.get_level_values("Peptide").unique(),
                               key=peptide_ranks.get, reverse=True)
    except KeyError as e:
        raise e  # Not having peptides is a series issue!

    else:
        hues = {peptide_order[i]:default_cycle[i % len(default_cycle)]
                for i in range(len(peptide_order))}
        hue_level_name = "Peptide"

    # Size per concentration or T Cell Number
    if "Concentration" in df_feature.index.names:
        size_level_name = "Concentration"
        if "TCellNumber" in df_feature.index.names:
            if df_feature.index.get_level_values("TCellNumber").unique():
                raise ValueError("Give a single TCellNumber if many concentrations")

    elif "TCellNumber" in df_feature.index.names:
        size_level_name = "TCellNumber"
    else:
        size_level_name = ""

    try:
        size_order = sorted(df_feature.index.get_level_values(size_level_name).unique(),
                           key=read_conc)
    except KeyError as e:
        sizes = dict()
        print("No size variable found")

    else:
        sizes_list = [maxwidth - maxwidth*0.75 * i / len(size_order) for i in range(len(size_order))]
        sizes_list = sizes_list[::-1]
        sizes = {size_order[i]: sizes_list[i] for i in range(len(size_order))}

    # Style per processing type
    styles_defaults = ['-', '--', '-.', ':']
    styles_ref = ["Fit", "Splines"]
    style_level_name = "Processing type"
    styles = dict()
    for s in styles_ref:
        if s not in df_feature.index.get_level_values(style_level_name).unique():
            styles = dict()
            style_level_name = ""
            break
        else:
            styles[s] = styles_defaults[styles_ref.index(s)]

    return [
        [hue_level_name, hues],
        [size_level_name, sizes],
        [style_level_name, styles]
    ]

def timecourse_smallplots(df_feature, feat_name="N", maxwidth=2., do_leg=True, **kwargs):
    """ Plot the time course vs fits for all peptides of the feature in df_feature
    (either N_i, n_i, or derivative) on two plots, one per node. Color
    per peptide, size per concentration. Make sure there's a single T cell number
    and not too many conditions. Works only on WT data for now.

    Returns:
        fig: the Figure
        [ax1, ax2, axleg, leg]
    """
    legend_info = create_labeling(df_feature, maxwidth=maxwidth)
    hue_level_name, hues = legend_info[0]
    size_level_name, sizes = legend_info[1]
    style_level_name, styles = legend_info[2]

    ## Plotting!
    if do_leg:
        gs = GridSpec(nrows=2, ncols=4)
    else:
        gs = GridSpec(nrows=2, ncols=3)
    fig = plt.figure()
    ax1 = fig.add_subplot(gs[0, :3])
    ax2 = fig.add_subplot(gs[1, :3], sharex=ax1)
    df_feature = df_feature.unstack("Time")
    index_names = list(df_feature.index.names)
    hue_pos = index_names.index(hue_level_name)
    if size_level_name != "":
        size_pos = index_names.index(size_level_name)
    else:
        size_pos = None
    if style_level_name != "":
        style_pos = index_names.index(style_level_name)
    else:
        style_pos = None

    # Plot each line
    for key in df_feature.index:
        times = df_feature.loc[key].index.get_level_values("Time").unique()
        times = sorted(times, key=float)
        times_f = np.asarray(times)
        n1_vals = df_feature.loc[key, "Node 1"].loc[times]
        n2_vals = df_feature.loc[key, "Node 2"].loc[times]
        # Load the color, linestyle, size corresponding to this key
        hue = hues.get(key[hue_pos])
        siz = sizes.get(key[size_pos]) if size_pos is not None else 2.
        sty = styles.get(key[style_pos]) if style_pos is not None else "-"
        ax1.plot(times_f, n1_vals, color=hue, lw=siz, ls=sty)
        ax2.plot(times_f, n2_vals, color=hue, lw=siz, ls=sty)

    # Label axes, add a legend, etc.
    if do_leg:
        axleg = fig.add_subplot(gs[:, 3:])
        axleg.set_axis_off()
        leg = add_hue_size_style_legend(
            axleg, hues, sizes, styles, size_level_name,
            style_level_name, hue_sort_key=peptide_ranks.get, **kwargs)
    else:
        leg = None
        axleg = None
    ax2.set_xlabel("Time [h]", size=8)
    ax1.set_ylabel(r"${}_1(t)$ [-]".format(feat_name), size=8)
    ax2.set_ylabel(r"${}_2(t)$ [-]".format(feat_name), size=8)
    ax1.tick_params(which="both", length=1.5, width=0.5, labelsize=6.)
    ax2.tick_params(which="both", length=1.5, width=0.5, labelsize=6.)
    return [fig, [ax1, ax2, axleg, leg]]


def latentspace_smallplot(df_feature, feat_name="N", maxwidth=2., do_leg=True, **kwargs):
    """ Plot the the two latent space nodes against each other
    (either N_i, n_i, or derivative). Color per peptide, size per
    concentration. Make sure there's a single T cell number
    and not too many conditions. Works only on WT data for now.

    Returns:
        fig: the Figure
        [ax1, axleg, leg]
    """
    legend_info = create_labeling(df_feature, maxwidth=maxwidth)
    hue_level_name, hues = legend_info[0]
    size_level_name, sizes = legend_info[1]
    style_level_name, styles = legend_info[2]

    ## Plotting!
    if do_leg:
        gs = GridSpec(nrows=1, ncols=5)
    else:
        gs = GridSpec(nrows=1, ncols=3)
    fig = plt.figure()
    ax1 = fig.add_subplot(gs[0, :3])
    df_feature = df_feature.unstack("Time")
    index_names = list(df_feature.index.names)
    hue_pos = index_names.index(hue_level_name)
    if size_level_name != "":
        size_pos = index_names.index(size_level_name)
    else:
        size_pos = None
    if style_level_name != "":
        style_pos = index_names.index(style_level_name)
    else:
        style_pos = None

    # Plot each line
    for key in df_feature.index:
        times = df_feature.loc[key].index.get_level_values("Time").unique()
        times = sorted(times, key=float)
        n1_vals = df_feature.loc[key, "Node 1"].loc[times]
        n2_vals = df_feature.loc[key, "Node 2"].loc[times]
        # Load the color, linestyle, size corresponding to this key
        hue = hues.get(key[hue_pos])
        siz = sizes.get(key[size_pos]) if size_pos is not None else 2.
        sty = styles.get(key[style_pos]) if style_pos is not None else "-"
        ax1.plot(n1_vals, n2_vals, color=hue, lw=siz, ls=sty)

    # Label axes, add a legend, etc.
    if do_leg:
        axleg = fig.add_subplot(gs[:, 3:])
        axleg.set_axis_off()
        leg = add_hue_size_style_legend(
            axleg, hues, sizes, styles, size_level_name,
            style_level_name, **kwargs)
    else:
        leg = None
        axleg = None
    ax1.set_xlabel(r"${}_1(t)$ [-]".format(feat_name), size=8)
    ax1.set_ylabel(r"${}_2(t)$ [-]".format(feat_name), size=8)
    ax1.tick_params(which="both", length=1.5, width=0.5, labelsize=6.)
    return [fig, [ax1, axleg, leg]]


### FOR PARAMETER PLOTS
def add_hue_size_style_scatter_legend(ax, hues, sizes, styles, sizes_name, styles_name,
                                hue_sort_key=None, **kwargs):
    """ Add a legend in the given ax or fig containing the existing handles (expected
        to be solid lines) and sections explaining the types of markers

        Tip: use kwargs to pass in small fontsize and handlelength,
            for Science figure format.
    Args:
        hues (dict): where keys are labels controlling hues, values are
            associated colors
        sizes (dict): where keys are labels controlling sizes, values are
            associated line widths
        styles (dict): where keys are labels controlling marker styles,
            values are associated line styles.
        sizes_name (str): name of the feature controlling size
        styles_name (str): name of the feature controlling style
        kwargs: keyword arguments passed directyl to ax.legend, like
            handlelength and fontsize

    Returns:
        matplotlib.legend.Legend: the legend created on the ax.
    """
    # Hues: no subtitle, this is the kind of lines expected.
    # Make solid lines of width equal to the max in sizes
    handles = []
    if len(sizes) > 0:
        maxwidth = max(sizes.values())
    else:
        maxwidth = 2.

    if hue_sort_key is not None:
        sorted_hue_keys = sorted(hues.keys(), key=hue_sort_key, reverse=True)
    else:
        sorted_hue_keys = hues.keys()

    for k in sorted_hue_keys:
        handles.append(Line2D([0], [0], label=k, markerfacecolor=hues[k], markeredgecolor=hues[k],
                              linestyle="none", markersize=maxwidth, marker="o"))

    # Line sizes: it could be an empty dict if there is a single size and
    # we do not want it in the legend
    if len(sizes) > 0:
        handles.append(LegendSubtitle(sizes_name))
        sorted_keys = sorted(sizes.keys(), key=read_conc, reverse=True)
        for k in sorted_keys:
            handles.append(Line2D([0], [0], label=k, marker="o",
                                  markerfacecolor="w", markeredgecolor="k",
                                  linestyle="none", markersize=sizes[k]))

    # Line styles
    if len(styles) > 0:
        handles.append(LegendSubtitle(styles_name))
        sorted_keys = list(styles.keys())
        if "-" in sorted_keys:
            sorted_keys.remove("-")
            sorted_keys.insert(0, "-")
        for k in sorted_keys:
            handles.append(Line2D([0], [0], label=k, marker=styles[k],
                                 linestyle="none", linewidth=maxwidth,
                                 markerfacecolor="w", markeredgecolor="k"))

    leg = ax.legend(handles=handles,
              handler_map={LegendSubtitle: LegendSubtitleHandler()},
              loc="upper left", bbox_to_anchor=(0, 1), **kwargs)
    return leg

def create_labeling_scatter(df_p, hue_level_name=None, hue_order=None, hue_map=None, size_level_name=None,
                            size_order=None, maxsize=6., style_level_name=None, style_order=None):
    """ Prepare color, style, size info for the keys in the specified
    level for each attribute.

    Returns:
        hue_info (list): lists of triplets of elements:
            [hue_level_name, hue_order, hues],
            [size_level_name, size_order, sizes],
            [style_level_name, style_order, styles]
    """
    if isinstance(df_p.index, pd.MultiIndex):
        index_names = df_p.index.names
    else:
        index_names = [df_p.index.name]

    # Hues
    if hue_level_name is None:
        hues = dict()
        hue_order = []
    else:
        if hue_map is None:
            hue_map = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if hue_order is None:
            hue_order = df_p.index.get_level_values(hue_level_name).unique()
        nhues = len(hue_map)
        hues = {hue_order[i]:hue_map[i % nhues] for i in range(len(hue_order))}

    # Sizes
    if size_level_name is None:
        sizes = dict()
        size_order = []
    else:
        if size_order is None:
            size_order = df_p.index.get_level_values(size_level_name).unique()
        nsizes = len(size_order)
        size_map = [0.25*maxsize + 0.75*maxsize*(i+1)/nsizes for i in range(nsizes)]
        sizes = {size_order[i]:size_map[i] for i in range(nsizes)}

    # Styles
    if style_level_name is None:
        styles = dict()
        style_order = []
    else:
        if style_order is None:
            style_order = df_p.index.get_level_values(style_level_name).unique()
        style_map = Line2D.filled_markers
        nstyles = len(style_map)
        styles = {style_order[i]:style_map[i] for i in range(len(style_order))}

    return [
        [hue_level_name, hue_order, hues],
        [size_level_name, size_order, sizes],
        [style_level_name, style_order, styles]
    ]

def paramspace_smallplots(df_p, hue_level_name=None, style_level_name=None, hue_map=None,
                          size_level_name=None, do_leg=True, maxsize=6., **kwargs):
    """ Function to plot parameters against each other.

    kwargs: keyword arguments passed to axes.legend
    """
    if not isinstance(df_p.index, pd.MultiIndex):
        raise TypeError("Assuming a MultiIndex for the index")
    if isinstance(df_p.columns, pd.MultiIndex):
        raise TypeError("Assuming a simple Index for the columns")
    index_names = df_p.index.names
    # Check which levels are present
    for n in index_names:
        if n not in ["Peptide", "Concentration", "Data", "TCellNumber", 'Time', 'Processing type']:
            raise KeyError("I can't deal with level {}".format(n))

    # Create order for peptides, concentrations, T cell numbers if necessary
    orders = dict()
    for name in [hue_level_name, style_level_name, size_level_name]:
        if name == "Peptide":
            orders["Peptide"] = sorted(df_p.index.get_level_values("Peptide").unique(),
                                   key=peptide_ranks.get, reverse=True)
        elif name == "Concentration":
            orders["Concentration"] = sorted(df_p.index.get_level_values("Concentration").unique(),
                                   key=read_conc)
        elif name == "TCellNumber":
            orders["TCellNumber"] = sorted(df_p.index.get_level_values("TCellNumber").unique(),
                                   key=read_conc)
        else:
            orders[name] = None

    # Now, create the labeling info
    hue_order = orders.get(hue_level_name, None)
    size_order = orders.get(size_level_name, None)
    style_order = orders.get(style_level_name, None)

    legend_info = create_labeling_scatter(df_p,
                            hue_level_name=hue_level_name, hue_order=hue_order, hue_map=hue_map,
                            size_level_name=size_level_name, size_order=size_order,
                            style_level_name=style_level_name, style_order=style_order,
                            maxsize=maxsize)

    hue_level_name, hue_order, hues = legend_info[0]
    size_level_name, size_order, sizes = legend_info[1]
    style_level_name, style_order, styles = legend_info[2]

    # Prepare the plot's axes, lower triangular layout
    # Use the upper triangular part for the legend
    params_to_plot = [lb for lb in df_p.columns if not lb.startswith("var")]
    # Loop over pairs of parameters, column-major, so the first column
    # parameter gets the most plots (first column, nparams-1 rows)
    fig = plt.figure()
    n_par = len(params_to_plot)
    gs = fig.add_gridspec(n_par - 1, n_par - 1)
    axes = [[None for i in range(n_par - 1)] for j in range(n_par - 1)]
    param_pairs = []
    idx_pairs = []
    for j in range(n_par-1):
        for i in range(j, n_par-1):
            # Pair of parameters corresponding to that subplot
            param_pairs.append((params_to_plot[j], params_to_plot[i+1]))
            idx_pairs.append((i, j))
            if i > j and j > 0:
                axes[i][j] = fig.add_subplot(gs[i, j], sharex=axes[j][j], sharey=axes[i][0])
            elif i > j:
                axes[i][j] = fig.add_subplot(gs[i, j], sharex=axes[j][j])
            elif j > 0:
                axes[i][j] = fig.add_subplot(gs[i, j], sharey=axes[i][0])
            else:
                axes[i][j] = fig.add_subplot(gs[i, j])  # This is just the 0, 0 case...

    # Combine grid cells to make space for the legend, as much as possible
    # Number of rows in the upper triangular, diagonal excluded = n_par - 2
    if n_par - 2 == 0:
        # Case where the anti-diagonal has one upper element, so 1*2 > 1*1
        axleg = fig.add_subplot(1, 2, 2)  # Use all the last column
    elif (n_par - 2) % 2 == 1:
        # Use the antidiagonal square
        antid_size = (n_par - 2) // 2 + 1
        axleg = fig.add_subplot(gs[:antid_size, -antid_size:])
    else:
        # Use the antidiagonal square plus one row below
        antid_size = (n_par - 2) // 2
        axleg = fig.add_subplot(gs[:antid_size+1, -antid_size:])

    # Prepare 1d arrays of the color and size of each point
    alpha = 0.7
    if hue_level_name is not None:
        cl = [(*to_rgb(a), alpha)
                for a in df_p.index.get_level_values(hue_level_name).map(hues)]
        cl = np.asarray(cl)
    else:
        cl = None
    if size_level_name is not None:
        sz = 0.25*df_p.index.get_level_values(size_level_name).map(sizes)**2
    else:
        sz = None

    for ii, pair in enumerate(param_pairs):
        xvals = df_p[pair[0]]
        yvals = df_p[pair[1]]
        i, j = idx_pairs[ii]
        # We need to plot one marker style at a time, slicing only
        # points with that style key (styk)
        if style_level_name is not None:
            for styk in styles.keys():
                sty = styles.get(styk)
                where_sty = np.asarray(
                    (df_p.index.get_level_values(style_level_name) == styk), dtype=bool)
                sz_sty = sz[where_sty] if size_level_name is not None else sz
                cl_sty = cl[where_sty] if hue_level_name is not None else cl
                axes[i][j].scatter(xvals[where_sty], yvals[where_sty], s=sz_sty,
                                c=cl_sty, marker=sty)
        else:
            axes[i][j].scatter(xvals, yvals, s=sz, c=cl)
        # Label this plot
        if i == n_par - 2:
            axes[i][j].set_xlabel(r"${}$".format(pair[0]), size=8, labelpad=0.5)
        if j == 0:
            axes[i][j].set_ylabel(r"${}$".format(pair[1]), size=8, labelpad=0.5)
        axes[i][j].tick_params(which="both", length=2., width=0.5, labelsize=6.)

    # Add a legend, etc.
    if do_leg:
        axleg.set_axis_off()
        if "ncol" not in kwargs.keys():
            kwargs["ncol"] = (n_par - 2)//2
        leg = add_hue_size_style_scatter_legend(
            axleg, hues, sizes, styles, size_level_name,
            style_level_name, **kwargs)
    else:
        leg = None
        axleg = None
    return [fig, [axes, axleg, leg]]


def barplots_levels(df, hue_lvl="Model", x_lvl="TCellNumber", groupwidth=0.7,
                    hue_map={}, hue_order=[], hue_reverse=True, x_reverse=True):
    """ PLot a row of barplots, one per column.
    x_lvl gives the level specifying groups of a given color,
    hue_lvl gives the level specifying which color of bar to plot.
    """
    # Prepare the labels for the hue and xaxis levels
    hue_groups = df.index.get_level_values(hue_lvl).unique()
    x_ticks = sorted(list(df.index.get_level_values(x_lvl).unique()),
                     key=read_conc, reverse=x_reverse)
    if hue_order != []:
        hue_groups = sorted(hue_groups, key=hue_order.index, reverse=hue_reverse)
    else:
        hue_groups = sorted(hue_groups, reverse=hue_reverse)

    # Custom function to slice at the proper levels, based on the indices of each level
    idx_hue_lvl, idx_x_lvl = df.index.names.index(hue_lvl), df.index.names.index(x_lvl)
    def poly_slice(x, h):
        fullslice = [slice(None)]*len(df.index.names)
        fullslice[idx_x_lvl] = x
        fullslice[idx_hue_lvl] = h
        return tuple(fullslice)

    # Prepare a hue map, if necessary
    if hue_map is None:
        hue_map = {hue_groups[i]:c for i, c in enumerate(sns.color_palette(n_colors=len(hue_groups)))}

    # Prepare the x axis of each subplot (same for all)
    assert groupwidth <= 1., "Can't have group width larger than 1"
    barwidth = groupwidth / len(hue_groups)
    # Position of the bars we will have
    x_leftbar_axis = np.arange(-groupwidth/2 + barwidth/2, len(x_ticks) - groupwidth/2 - barwidth/2, 1.)

    # Prepare plot
    fig = plt.figure()
    frac_legw = 2
    gs = fig.add_gridspec(nrows=1, ncols=df.shape[1]*frac_legw + 1)

    # Loop over features
    axes = []
    for i, feat in enumerate(df.columns):
        ax = fig.add_subplot(gs[0, frac_legw*i:frac_legw*(i+1)])
        axes.append(ax)
        # Loop over hues, because they correspond to different bars at
        # distinct x positions.
        for j, h in enumerate(hue_groups):
            ax.bar(x_leftbar_axis+barwidth*j, df.loc[poly_slice(x_ticks, h), feat],
                         align="center", width=barwidth, color=hue_map.get(h, None), label=h)

        # Label this plot
        try:
            ax.set_ylabel(feat[-1], fontsize=8)
        except TypeError:
            ax.set_ylabel(feat, fontsize=8)
        #ax.set_xlabel(x_lvl, fontsize=8)
        ax.set_xticks(range(len(x_ticks)))
        ax.set_xticklabels(x_ticks, fontsize=7)
        ax.tick_params(axis="both", width=1., length=3., labelsize=7)


    # Add legend
    legax = fig.add_subplot(gs[0, -1])
    legax.set_axis_off()
    legax.legend(*ax.get_legend_handles_labels(), loc="upper left",
                 bbox_to_anchor=(0, 0.8), frameon=False, fontsize=8, borderaxespad=-2)
    return fig, axes, legax


def add_legend_subtitles_huemaps(subtitles, hue_maps, hue_levels_order, ax, **kwargs):
    """ Add a legend to axes ax for a list of models (subtitles in the legend)
    and a list of hue maps (dictionaries), where keys are labels and values are colors.
    All kwargs are passed to legend()
    """
    # Create the subtitles and patches
    handles = []
    for i in range(len(subtitles)):
        handles.append(LegendSubtitle(subtitles[i]))
        # Add a patch for each entry in the ith hue map
        hue_levels = sorted(list(hue_maps[i].keys()), key=hue_levels_order.index)
        for lbl in hue_levels:
            handles.append(mpatches.Patch(color=hue_maps[i][lbl], label=lbl))

    # Add the legend

    ncol = kwargs.get("ncol", len(subtitles))
    kwargs["ncol"] = ncol
    leg = ax.legend(handles=handles,
              handler_map={LegendSubtitle: LegendSubtitleHandler()},
              **kwargs)
    return leg
