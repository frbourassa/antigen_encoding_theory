# Antigen encoding from high dimensional cytokine dynamics -- theory
Repository with all the necessary code for modelling cytokine dynamics in
latent space, generating back cytokine data with reconstruction from the
latent space, and computing the channel capacity of cytokine dynamics for
mutual information with antigen quality. The code produces all the
necessary results to reproduce the main and supplementary figures related to
theoretical parts (latent space modelling and channel capacity) of the paper
> Achar, S., Bourassa, F.X.P., Rademaker, T., ..., François, P., and Altan-Bonnet, G.,
"Universal antigen encoding of T cell activation from high dimensional cytokine data",
submitted, 2021.

All data necessary to run the code is hosted at:
> TBD


## Installation
This project depends on code modules hosted as separate projects on Github and Gitlab and included as git submodules. We proceeded this way because we use thosemodules in various research projects and it was easier to update them everywhere like that. Therefore, to obtain all necessary submodules for this repository's code to run, clone the repo using the `--recurse-submodules` option of `git clone`:
```
git clone --recurse-submodules https://github.com/frbourassa/antigen_encoding_theory.git
```
For help with projects that contain git submodules, see the
[`git submodule.  man page](https://git-scm.com/book/en/v2/Git-Tools-Submodules).

## Contents
The most important calculations are in Jupyter notebooks (Python language):

- `fit_ballistic_model.ipynb`: fit models (constant velocity and force with matching) to cytokine time integrals projected in latent space. Some results of this notebook are used in main figure 3 and supplementary figures related to model fits and model parameter space (supp. section 4).

- `reconstruct_cytokines_fromLSdata.ipynb`: project multi-dimensional cytokine time integrals into latent space, optimize and test a reconstruction (i.e., decoder) model that recovers the full cytokine time courses from latent space trajectories. The results of this notebook are used in main figure 2 and supplementary figures related to cytokine reconstruction (supp. section 5)

- `reconstruct_cytokines_fromLSmodel_pvalues.ipynb`: use latent space models and reconstruction as a model that fits experimental cytokine time series (supp. section 5). Project multi-dimensional cytokine time integrals into latent space, fit model trajectories in latent space, and reconstruct the cytokine trajectories corresponding to the model fits using a pre-optimized reconstruction model. Produces directly the supplementary figure about using the latent space model as a cytokine model.

- `generate_synthetic_data.ipynb`: use latent space models and reconstruction as a generative model of cytokine time series (supp. section 5). Project data fit model parameters on latent space trajectories; then, estimate the distributions of model parameters for each antigen, sample from them, and reconstruct the cytokine trajectories corresponding to the chosen parameter values, using a pre-optimized reconstruction model. Results of this notebook are used for the supplementary figure about generation of synthetic cytokine time series.

-`TBD`: 

Many of the calculations performed in the above notebooks rely on lower-level functions defined in sub-modules. The code is documented in-place with comments, and explained in the supplementary text of the paper. These modules are:
- `ltspcyt`: code to process (smooth and interpolate) raw cytokine dataframes, import processed data, fit latent space models, reconstruct cytokines from latent space trajectories.
- `chancapmc`: C code (wrapped with the Python C-API) to compute channel capacity between antigen quality and model parameters describing the corresponding latent space trajectories.


Other notebooks import the results saved by the notebooks above to create figures included in the main text and supplementary information. The code for figures was kept separate from the bulk of calculations because some figures require a lot of matplotlib commands! These plotting notebooks are:
- `TBD`:
- `TBD`:

More calculations for specific figures and supplementary figures are found in other Python scripts in the `more_main_scripts/` folder:
- `manifold_dimension.py`: calculation of the cytokine manifold Hausdorff dimension from the correlation function scaling
- `projection_3d_movie.py`: code to generate animated three-dimensional graphs of time courses of cytokine concentrations and time integrals.
