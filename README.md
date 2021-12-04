# Antigen encoding from high dimensional cytokine dynamics – theory

Repository with all the necessary code for modelling cytokine dynamics in
latent space, generating back cytokine data with reconstruction from the
latent space, and computing the channel capacity of cytokine dynamics for
mutual information with antigen quality. The code generates all the
necessary results to reproduce the main and supplementary figures related to theoretical parts (latent space modelling and channel capacity) of the paper
> Achar, S., Bourassa, F.X.P., Rademaker, T., ..., François, P., and Altan-Bonnet, G.,
"Universal antigen encoding of T cell activation from high dimensional cytokine data",
submitted, 2021.

Weights of the neural network that produces the latent space used throughout the paper are provided, along with all data necessary to run the code. More neural networks can be trained and more cytokine data processing and fitting can be done using the cytokine-pipeline user interface,
[also hosted on Github](https://github.com/soorajachar/antigen-encoding-pipeline).



## Installation

### Compilation of the ``chancapmc`` C module
You need to run the Python setup script: first navigate to the chancapmc/ folder, then execute the script and move the built library (.so file) in the chancapmc/ folder:
```bash
>>> cd chancapmc/
>>> python setup_chancapmc.py build_ext --inplace
>>> mv build/lib*/chancapmc*.so .
```
Then, if you want, try running the test file for the Python interface:
```bash
python test_chancapmc.py
```
Unit tests in C are also available in the script `unittests.c`. Compile and execute them according to the instructions given in the header of the file.

More details can be found in the Github repository where this module is hosted separately: https://github.com/frbourassa/chancapmc .

### Downloading data
Data is tracked in the git repository and is downloaded with the code when the repository is cloned. There are starting data files in four folders:

 - `data/initial/`: Contains the raw cytokine time series data.
 - `data/LOD`: Contains limits of detection used to process some files (this one is not essential, the code can run without it).
 - `data/misc/`: Contains plotting parameters and EC50 values for antigens.
 - `data/trained-networks/`: Contains projection matrix and normalization factors to construct the latent space, and also other weight matrices of the default neural network.

Alternatively, if you want to download a version of the code without the data, or make sure the data you have is correct, look at release `v0.2.0-alpha`. The data files are then available as zip files from the Github release notes of [v0.2.0-alpha](https://github.com/frbourassa/antigen_encoding_theory/releases/tag/v0.2.0-alpha). But the official, up-to-date version of the code is currently `v1.0.0`, which includes data files in the git history.
 - ``antigen_encoding_cytokine_timeseries.zip``: unzip contents in ``data/initial/``.
 - ``antigen_encoding_cytokine_lods.zip``: unzip contents in ``data/LOD/``.
 - ``antigen_encoding_data_misc.zip``: unzip contents in ``data/misc/``.
 - ``antigen_encoding_trained_networks.zip``: unzip contents in ``data/trained-networks/``.


### Data preprocessing
After downloading the git repository along with the data (or unzipping the cytokine data files (HDF5 format) in ``data/initial/``, run the script ``run_first_prepare_data.py``, which will save cleaned up versions of the raw dataframes in ``data/final/``, then process all cytokine time series (log transformation, smoothing spline interpolation, time integration), and save the processed time series in ``data/processed/``.

---
**IMPORTANT**

Run ``run_first_prepare_data.py`` the first time you download the code and data. Otherwise no analysis script will run.

---


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
The following diagram represents the main dependencies between most of the scripts in this project. Scripts are colored per theme (neural networks: pale orange, model fitting: orange, reconstruction: yellow, channel capacity: green, data processing: pink).  Indented scripts are those which need other scripts to be run first, as indicated by arrows on the left or right, annotated with the folders where intermediate results are stored. Scripts which produce figures included in the main or supplementary text are indicated by arrows going to the sub-folders in `figures/`.

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

Moreover, a C compiler is necessary to build the module ``chancapmc`` (C code interfaced with the Python C API). The module was tested with compilers Apple ``clang 11.0.3`` (macOS) and GNU ``gcc 4.9.4`` (Linux).  

## Note on sub-modules
The code modules `ltspcyt` (<ins>la</ins>tent <ins>sp</ins>ace <ins>cyt</ins>okines) and `chancapmc` (<ins>chan</ins>nel <ins>cap</ins>acity <ins>M</ins>onte <ins>C</ins>arlo) contain the core functions for data processing, latent space building, model fitting (`ltspcyt`), and channel capacity calculation (`chancapmc`).

The `ltspcyt` module is basically is a collection of the core functions under the GUI of [antigen-encoding-pipeline](https://github.com/soorajachar/antigen-encoding-pipeline), with added functions and classes for cytokine reconstruction.  Of interest, it includes  a customized version of *Scipy*'s  `curve_fit` function that is applicable to vector-valued functions of a scalar variable and can add L1 regularization of the fitted parameters.

`chancapmc` is also hosted separately on [Github](https://github.com/frbourassa/chancapmc): https://github.com/frbourassa/chancapmc It is licensed under the more permissive BSD 3-Clause-License.

This module provides functions to calculate channel capacity between any discrete input variable Q and continuous (vectorial) output variable **Y** which has multivariate normal conditional distributions P(**Y**|Q). It may be extended to more multivariate distributions in the future.


## License information
This repository is licensed under the GNU GPLv3.0 because one of the scripts (`estimate_channel_capacity_cce.ipynb`) uses the [`channel-capacity-estimator` package](https://github.com/pawel-czyz/channel-capacity-estimator)
from [Grabowksi et al., 2019](https://dx.doi.org/10.1098/rsif.2018.0792),
which is also licensed under GPLv3.0. Other dependencies are licensed under the BSD-3-Clause License, which is compatible with GPLv3.0.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5758462.svg)](https://doi.org/10.5281/zenodo.5758462)
