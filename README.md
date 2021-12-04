# Antigen encoding from high dimensional cytokine dynamics – theory

Repository with all the necessary code for modelling cytokine dynamics in
latent space, generating back cytokine data with reconstruction from the
latent space, and computing the channel capacity of cytokine dynamics for
mutual information with antigen quality. The code generates all the
necessary results to reproduce the main and supplementary figures related to
theoretical parts (latent space modelling and channel capacity) of the paper
> Achar, S., Bourassa, F.X.P., Rademaker, T., ..., François, P., and Altan-Bonnet, G.,
"Universal antigen encoding of T cell activation from high dimensional cytokine data",
submitted, 2021.

Weights of the neural network that produces the latent space used throughout the paper are provided, along with all data necessary to run the code. More neural networks can be trained and more cytokine data processing and fitting can be done using the cytokine-pipeline user interface,
[also hosted on Github](https://github.com/soorajachar/antigen-encoding-pipeline).



## Installation

### Downloading the repository and submodules
This project depends on code modules hosted as separate projects on Github and Gitlab and included as git submodules. Therefore, to obtain all necessary submodules for this repository's code to run, clone the repo using the `--recurse-submodules` option of `git clone`:
```
git clone --recurse-submodules https://github.com/frbourassa/antigen_encoding_theory.git
```
For help with projects that contain git submodules, see the
[`git submodule` man page](https://git-scm.com/book/en/v2/Git-Tools-Submodules).

### Compilation of the ``chancapmc`` C module


### Downloading data
Four compressed data files need to be unzipped in the appropriate folders before running the code.
 - ``antigen_encoding_cytokine_timeseries.zip``: unzip contents in ``data/initial/``. Contains the time series data.
 - ``antigen_encoding_cytokine_lod.zip``: unzip contents in ``data/LOD/``. Contains limits of detection used to process some files (this one is not essential, the code can run without it).
 - ``antigen_encoding_data_misc.zip``: unzip contents in ``data/misc/``. Contains plotting parameters and EC50 values for antigens.
 - ``antigen_encoding_network_weights.zip``: unzip contents in ``data/trained-networks/``. Contains projection matrix and normalization factors to construct the latent space, and also other weight matrices of the default neural network.

They are available on demand from the authors. Also available on demand are the output files of the code, for users having trouble running the code themselves.

### Data preprocessing
After downloading and unzipping the cytokine data files (HDF5 format) in ``data/initial/``, run the script ``run_first_prepare_data.py``, which will save cleaned up versions of the raw dataframes in ``data/final/``, then process all cytokine time series (log transformation, smoothing spline interpolation, time integration), and saved the processed time series in ``data/processed/``.



## Suggested order in which to run the code

As a general rule, always run scripts from the top folder of the repository (i.e. folder ``antigen_encoding_theory/`` unless you renamed it) and not from subfolders within.

Some scripts and Jupyter notebooks depend on the outputs saved to disks by other code files. The first time you use this repository, we suggest executing them in the following order. Afterwards, once the outputs are saved on disk, it becomes easier to go from one code file to the other.

 0. `run_first_prepare_data.py` to process data.
 1. `fit_latentspace_model.ipynb` to fit model parameters on latent space trajectories. Ideally, run three times, once for each version of the model.
 2. `reconstruct_cytokines_fromLSdata.ipynb` to reconstruct cytokine time series from data projected in latent space.
 3. `reconstruct_cytokines_fromLSmodel_pvalues.ipynb` to use the latent space model as a cytokine model that can fit cytokines themselves, via reconstruction.
 4. `generate_synthetic_data.ipynb` to generate new cytokine time series by sampling model parameters and reconstructing the corresponding cytokines.
 5. `compute_channel_capacity_HighMI_3.ipynb` to compute channel capacity $C$ of antigen encoding using interpolation of multivariate gaussian distributions in parameter space and our `chancapmc` module (Blahut-Arimoto algorithm with Monte Carlo integration for continuous input-output pdfs).
 6. `theoretical_antigen_classes_from_capacity_HighMI_3.ipynb` to subdivide the continuum of antigen qualities into $2^{C}$ ''theoretical'' antigen classes.

Once these main codes are run and their outputs saved, secondary scripts can be run more freely, and lastly plotting functions
 7. Secondary calculations in `more_main_scripts/`. Some will save further output files used by plotting scripts.
 8. Finally, run plotting scripts in `main_plotting_scripts/`.

We give more details on these scripts and notebooks in the CONTENTS.md file. Most files also contain indications on their dependencies in their headers. Also, the flowchart below illustrates those dependencies and the folders in which they share files.



## Diagram of the code structure
The following diagram represents the main dependencies between scripts in this project. Scripts are colored per theme (model fitting: orange, reconstruction: yellow, channel capacity: green, data processing: pink).  Indented scripts are those which need other scripts to be run first, as indicated by arrows on the left or right, annotated with the folders where intermediate results are stored. Scripts which produce figures included in the main or supplementary text are indicated by arrows going to the folders `figures/main/` or `figures/supp/`.

![Code structure diagram](figures/code_chart_short.svg)


## Requirements
The Python code was tested on Mac (macOS Catalina 10.15.7) and Linux (Linux 3.2.84-amd64-sata x86_64) with the Anaconda 2020.07 distribution, with the following versions for important packages:
 - Python 3.7.6
 - numpy 1.19.2
 - scipy 1.5.2
 - pandas 1.2.0
 - matplotlib 3.3.2
 - seaborn 0.11.1
 - scikit-learn 0.23.2

The following additional Python packages were installed and are necessary for specific scripts in the project (but not needed for most code):
 - tensorflow 2.0.0 (macOS) or 2.3.0 (Linux)
 - wurlitzer 2.0.1
 - channel-capacity-estimator 1.0.1 (see [Github page](https://github.com/pawel-czyz/channel-capacity-estimator))

The exact Python configuration used is included in ``data/python_environment.yml``.

Moreover, a C compiler is necessary to build the module ``chancapmc`` (C code interfaced with the Python C API). The module was tested with compilers ``clang-1103.0.32.62`` (macOS) and ``gcc 4.9.4`` (Linux).  


## License information
This repository is licensed under the GNU GPLv3.0 because one of the scripts (`estimate_channel_capacity_cce.ipynb`) uses the `channel-capacity-estimator` package from Grabowksi et al., 2019, which is also licensed under GPLv3.0. Other dependencies are licensed under the BSD 3-clause license, which is compatible with GPLv3.0.
