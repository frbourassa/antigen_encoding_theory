# Contents: theoretical analysis of antigen encoding

## Main calculations
The most important calculations are in Jupyter notebooks (Python language):

- `fit_latentspace_model.ipynb`: fit models (constant velocity and force with matching) to cytokine time integrals projected in latent space. Some results of this notebook are used in main figure 3 and supplementary figures related to model fits and model parameter space (supp. section 4).

- `reconstruct_cytokines_fromLSdata.ipynb`: project multi-dimensional cytokine time integrals into latent space, optimize and test a reconstruction (i.e., decoder) model that recovers the full cytokine time courses from latent space trajectories. The results of this notebook are used in main figure 2 and supplementary figures related to cytokine reconstruction (supp. section 5)

- `reconstruct_cytokines_fromLSmodel_pvalues.ipynb`: use latent space models and reconstruction as a model that fits experimental cytokine time series (supp. section 5). Project multi-dimensional cytokine time integrals into latent space, fit model trajectories in latent space, and reconstruct the cytokine trajectories corresponding to the model fits using a pre-optimized reconstruction model. Produces directly the supplementary figure about using the latent space model as a cytokine model.

- `generate_synthetic_data.ipynb`: use latent space models and reconstruction as a generative model of cytokine time series (supp. section 5). Project data fit model parameters on latent space trajectories; then, estimate the distributions of model parameters for each antigen, sample from them, and reconstruct the cytokine trajectories corresponding to the chosen parameter values, using a pre-optimized reconstruction model. Results of this notebook are used for the supplementary figure about generation of synthetic cytokine time series.

- `compute_channel_capacity_HighMI_3.ipynb`: channel capacity calculation between antigen quality and model parameters describing latent space time courses. Results of this notebook are used for main and supplementary figures related to channel capacity and theoretical antigen classes.

- `theoretical_antigen_classes_from_capacity_HighMI_3.ipynb`: determining theoretical antigen classes from the channel capacity calculation and resulting optimal antigen distribution, plotting their latent space trajectories and model parameter space distributions, and even reconstructing the corresponding cytokine time series.  Directly produces main figure 3, panels C and D (maybe we should split it eventually).



## Secondary calculations
More secondary calculations and plotting used in specific figures and supplementary figures are found in other Python scripts in the `more_main_scripts/` folder:

- `cytokines_distribution_noise.ipynb`: study the distribution of cytokines in linear and log scale, calculate signal-to-noise ratio (SNR) of different cytokines. Generates supplementary figures about processing and SNR of IL-4 and IL-10.

- `including_IL-4_IL-10.ipynb`: show that including IL-4 and IL-10 in the classifier's inputs does not change the latent space. Produces the supplementary figure about these cytokines.

- `smoothing_log_integral_importance.ipynb`: study the impact of different preprocessing modalities on cytokine time series: logarithmic transform or not, time series smoothing or not, time integration or not. Produces the supplementary figure about the impact of preprocessing on the latent space.  

- `autoencoder_mi_cytokines.ipynb`: train an autoencoder on cytokine time integrals and compare the latent space obtained with this unsupervised method to the antigen encoding latent space and to PCA. Creates the supplementary figure about autoencoders.

- `manifold_dimension.py`: calculation of the cytokine manifold Hausdorff dimension from the correlation function scaling. Directly produces the supplementary figure about Hausdorff dimension.


- `paramspace_distance_drugs_withPCA.py`: compute Earth Mover's distance, or another distribution distance metric, between model parameter distributions for naive and drug-perturbed conditions, for each drug among the panel tested. Distributions are projected on PCA axes before distances are computed along each PC. Similar calculations are made for main figure 4. The distance functions are defined in the `metrics/` folder.

- `chancap_interpol_bootstrap.py`: script that runs multiple repeats of the channel capacity calculation, perturbing the regularization hyper-parameters used in model fitting as a way to assess the robustness of the channel capacity result against perturbations in the model parameter distributions. It uses multiprocessing for improved speed, but still takes many minutes to converge, depending on the number of repeats carried out.

- `reconstruct_cytokines_chancap_antigen_prototypes.ipynb`: generate prototypical cytokine time series for the theoretical antigen classes found in `theoretical_antigen_classes_from_capacity_HighMI_3.ipynb`, using a combination of the latent space model and cytokine reconstruction.

- `estimate_channel_capacity_cce.ipynb`: Estimate channel capacity of model parameter space using the  [algorithm of Grabowski et al., 2019](https://dx.doi.org/10.1098/rsif.2018.0792). To run this notebook, you first need to install the `cce` package, following the instructions given on its [Github page](https://github.com/pawel-czyz/channel-capacity-estimator).

- ``mi_timecourse_from_cytokines_and_model.ipynb``: calculate mutual information between antigen quality and cytokines or latent space variables over a sliding time window, like in figure 1 of the paper.

- `human_tcr_analysis.ipynb`: fits the constant velocity model (parameter $v_0$) on the cytokine time series of human T cells.



## Plotting code
Some of the code listed above produces panels included in the main text or supplementary text:
- `theoretical_antigen_classes_from_capacity_HighMI_3.ipynb`
- `reconstruct_cytokines_fromLSmodel_pvalues.ipynb`
- `more_main_scripts/manifold_dimension.py`

Other Jupyter notebooks import the results saved by the notebooks above, and sometimes perform minor supplementary calculations, to create figures included in the main text and supplementary information. They are in the `main_plotting_scripts/`folder. The code for those figures was kept separate from the bulk of calculations because the results could be exported easily and some figures require a lot of matplotlib commands. These plotting notebooks are:
- `spline_process_explanation.py`: to create a supplementary figure detailing the steps of processing, smoothing and interpolation that we apply to experimental cytokine time series.
- `plot_spline_panels.py`: plot cytokine time series and smoothing spline functions fitted on them for an entire dataset, with one panel per peptide and cytokine.
- `model_fits_supp_panels.ipynb`: to create supplementary figures about latent space model fits. Need to run `fit_latentspace_model.ipynb` first.
- `recon_supp_panels.ipynb`: to create supplementary figures about cytokine reconstruction and synthetic data generation. Need to run `reconstruct_cytokines_fromLSdata.ipynb` and `generate_synthetic_data.ipynb` first.
- `reconstruction_linear_example.py`: cartoon illustrating why the cytokine manifold can't be perfectly reconstruction with linear regression alone.
- `projection_3d_movie.py`: code to generate animated three-dimensional graphs of time courses of cytokine concentrations and time integrals.
- `mi_results_supp_panels.ipynb`: to create supplementary figures about mutual information and channel capacity. Need to run  `compute_channel_capacity_HighMI_3.ipynb` and `more_main_scripts/estimate_channel_capacity_cce.ipynb` first.
- `peptide_channel_diagrams.py`: to produce the supplementary figure cartoon explaining the channel capacity calculation procedure.
- `latentspace_weights_interpretation.ipynb`: output layer weights interpretation and interpolation at the EC50 values of theoretical antigen classes found from channel capacity results. Generates panels for the supplementary figure about the neural network's weights interpretation.


### Submodules hosted separately
Many of the calculations performed in the above notebooks rely on lower-level functions defined in sub-modules. The code is documented in-place with comments, and explained in the supplementary text of the paper. These modules are:

- `ltspcyt`: code to process (smooth and interpolate) raw cytokine dataframes, import processed data, fit latent space models, reconstruct cytokines from latent space trajectories.

- `chancapmc`: C code (wrapped with the Python C-API) to compute channel capacity between antigen quality and model parameters describing the corresponding latent space trajectories.



## Utility functions
Some relevant calculations are defined in the scripts of the `utils/` folder. The noteworthy ones are:

- `statistics.py`: contains statistical estimators of multivariate normal distributions. Also contains a homemade PCA implementation.

- `distrib_interpolation.py`: functions to interpolated multivariate normal distribution parameters (means and Cholesky matrices).

- `recon_scaling.py` : functions to scale back reconstructed cytokine trajectories to absolute cytokine concentration units.

- Other files mostly contain functions imported by scripts or notebooks in the `main_plotting_scripts/` folder to make supplementary figures.


A few other functions are in the `metrics/` folder. There is first a Python implementation of the MI estimator proposed by [(Kraskov, Stögbauer and Grassberger, 2004)](https://link.aps.org/doi/10.1103/PhysRevE.69.066138)
 and applied by [(Ross, 2014)](https://doi.org/10.1371/journal.pone.0087357) to discrete input, continuous output distribution (like in our case with antigens--cytokines).  
 There is also the Kendall tau metric for our order accuracy measure.
 Then, there are different metrics to compute the distance between two sample distributions. To compare drug-perturbed model parameter distributions to unperturbed ones, we used Earth Mover's distance, but other distances are made available here.

- `discrete_continuous_info.py`: our Python implementation of the bin-less MI estimator by Ross, 2014, made faster by using *Scipy*'s `cKDTree` class.

- `count_inversions.py`: script to compute the Kendall tau order metric of a list by counting the number of inversions in it, or equivalently the number of neighbor swaps required to order it.

- `figure4_metrics.py`: function to compute some distance between two distributions projected along one or multiple PCA axes.

- `earth_mover.py`: network-based calculation of Earth Mover's distance (EMD), useful when data is more than 1D. For the 1D case, the EMD reduces to the Wasserstein distance, which is implemented in *Scipy* as `scipy.stats.wasserstein_distance`.

- `kldiv.py`: Kullback-Leibler divergence estimator, [written by David Huard](https://mail.python.org/pipermail/scipy-user/2011-May/029521.html) and found in a [Github gist](https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518), with a slight modification by us to prevent bugs from arising when identical or excessively close samples are present.
