# Antigen encoding from high dimensional cytokine dynamics – theory

Repository with all the necessary code for modelling cytokine dynamics in
latent space, generating back cytokine data with reconstruction from the
latent space, and computing the channel capacity of cytokine dynamics for
mutual information with antigen quality. The code produces all the
necessary results to reproduce the main and supplementary figures related to
theoretical parts (latent space modelling and channel capacity) of the paper
> Achar, S., Bourassa, F.X.P., Rademaker, T., ..., François, P., and Altan-Bonnet, G.,
"Universal antigen encoding of T cell activation from high dimensional cytokine data",
submitted, 2021.

A neural network can be trained using the cytokine-pipeline user interface. Weights of the network weights used throughout the paper are provided along with all data necessary to run the code, [also hosted on Github](https://github.com/tjrademaker/cytokine-pipeline).


## Installation
This project depends on code modules hosted as separate projects on Github and Gitlab and included as git submodules. We proceeded this way because we use thosemodules in various research projects and it was easier to update them everywhere like that. Therefore, to obtain all necessary submodules for this repository's code to run, clone the repo using the `--recurse-submodules` option of `git clone`:
```
git clone --recurse-submodules https://github.com/frbourassa/antigen_encoding_theory.git
```
For help with projects that contain git submodules, see the
[`git submodule` man page](https://git-scm.com/book/en/v2/Git-Tools-Submodules).

## Contents

### Main calculations
The most important calculations are in Jupyter notebooks (Python language):

- `fit_latentspace_model.ipynb`: fit models (constant velocity and force with matching) to cytokine time integrals projected in latent space. Some results of this notebook are used in main figure 3 and supplementary figures related to model fits and model parameter space (supp. section 4).

- `reconstruct_cytokines_fromLSdata.ipynb`: project multi-dimensional cytokine time integrals into latent space, optimize and test a reconstruction (i.e., decoder) model that recovers the full cytokine time courses from latent space trajectories. The results of this notebook are used in main figure 2 and supplementary figures related to cytokine reconstruction (supp. section 5)

- `reconstruct_cytokines_fromLSmodel_pvalues.ipynb`: use latent space models and reconstruction as a model that fits experimental cytokine time series (supp. section 5). Project multi-dimensional cytokine time integrals into latent space, fit model trajectories in latent space, and reconstruct the cytokine trajectories corresponding to the model fits using a pre-optimized reconstruction model. Produces directly the supplementary figure about using the latent space model as a cytokine model.

- `generate_synthetic_data.ipynb`: use latent space models and reconstruction as a generative model of cytokine time series (supp. section 5). Project data fit model parameters on latent space trajectories; then, estimate the distributions of model parameters for each antigen, sample from them, and reconstruct the cytokine trajectories corresponding to the chosen parameter values, using a pre-optimized reconstruction model. Results of this notebook are used for the supplementary figure about generation of synthetic cytokine time series.

- `compute_channel_capacity_HighMI_13.ipynb`: channel capacity calculation between antigen quality and model parameters describing latent space time courses. Results of this notebook are used for main and supplementary figures related to channel capacity and theoretical antigen classes.

- `theoretical_antigen_classes_from_capacity_HighMI_13.ipynb`: determining theoretical antigen classes from the channel capacity calculation and resulting optimal antigen distribution, plotting their latent space trajectories and model parameter space distributions, and even reconstructing the corresponding cytokine time series.  Directly produces main figure 3, panels C and D (maybe we should split it eventually). 

- `estimate_channel_capacity_cce.ipynb`: Estimate channel capacity of model parameter space using the  [algorithm of Grabowski et al., 2019](https://dx.doi.org/10.1098/rsif.2018.0792). To run this notebook, you first need to install the `cce` package, following the instructions given on its [Github page](https://github.com/pawel-czyz/channel-capacity-estimator).

### Secondary calculations
More secondary calculations and plotting used in specific figures and supplementary figures are found in other Python scripts in the `more_main_scripts/` folder:

- `manifold_dimension.py`: calculation of the cytokine manifold Hausdorff dimension from the correlation function scaling. Directly produces the supplementary figure about Hausdorff dimension

- `latentspace_weights_interpretation.ipynb`: output layer weights interpretation and interpolation at the EC50 values of theoretical antigen classes found from channel capacity results. Generates panels for the supplementary figure about the neural network's weights interpretation.

- `chancap_interpol_bootstrap.py`: script that runs multiple repeats of the channel capacity calculation, perturbing the regularization hyper-parameters used in model fitting as a way to assess the robustness of the channel capacity result against perturbations in the model parameter distributions. It uses multiprocessing for improved speed, but still takes many minutes to converge, depending on the number of repeats carried out.

### Submodules hosted separately
Many of the calculations performed in the above notebooks rely on lower-level functions defined in sub-modules. The code is documented in-place with comments, and explained in the supplementary text of the paper. These modules are:
- `ltspcyt`: code to process (smooth and interpolate) raw cytokine dataframes, import processed data, fit latent space models, reconstruct cytokines from latent space trajectories.

-`chancapmc`: C code (wrapped with the Python C-API) to compute channel capacity between antigen quality and model parameters describing the corresponding latent space trajectories.

### Plotting code
Some of the notebooks listed above produce panels included in the main text or supplementary text, because it would have been uselessly cumbersome to save all their results to disk and re-import them in a separate plotting script:
- `theoretical_antigen_classes_from_capacity_HighMI_13.ipynb`
- `reconstruct_cytokines_fromLSmodel_pvalues.ipynb`
- `more_main_scripts/manifold_dimension.py`

Other Jupyter notebooks import the results saved by the notebooks above, and sometimes perform minor supplementary calculations, to create figures included in the main text and supplementary information. They are in the `main_plotting_scripts/`folder. The code for those figures was kept separate from the bulk of calculations because the results could be exported easily and some figures require a lot of matplotlib commands. These plotting notebooks are:
- `TBD`: main figures 1 to 4
- `TBD`: the various panels... notebooks in supp_code/
- `peptide_channel_diagrams.py`: to produce the supplementary figure cartoon explaining the channel capacity calculation procedure.
- `projection_3d_movie.py`: code to generate animated three-dimensional graphs of time courses of cytokine concentrations and time integrals.


## License information
This repository is licensed under GPL-3.0 because one of its scripts (`estimate_channel_capacity_cce.ipynb`) uses the `channel-capacity-estimator` package from Grabowksi et al., 2019, which is also licensed under GPL-3.0. Other dependencies are licensed under the BSD 3-clause license, which is compatible with GPL-3.0.
