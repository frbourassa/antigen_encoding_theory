"""
Script containing the necessary function for the cytokine reconstruction
pipeline, with custom classes for linear regression.

@author:frbourassa
July 2, 2020
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## sklearn imports
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.neighbors import KernelDensity

from ltspcyt.scripts.adapt_dataframes import sort_SI_column
from ltspcyt.scripts.sigmoid_ballistic import (sigmoid_conc_full_freealpha,
    ballistic_v0, ballistic_F, ballistic_sigmoid, ballistic_sigmoid_freealpha)

## Backup palette
palette_backup = plt.rcParams['axes.prop_cycle'].by_key()['color']
peptides_backup = ["N4", "Q4", "T4", "V4", "G4", "E1", "A2", "Y3", "A8"]

### Functions and classes for linear and quadratic regression

# Linear regression using the Moore-Penrose pseudo-inverse.
def lsq_inverse(X, Y, y0=None):
    """ Compute the inverse projection matrix Q that minimizes
    the least-squares difference between the original data X and the
    reconstruction Q(Y-y0) from the projected points Y = PX + y0.
    The analytical solution shows that Q = XY^+, where Y^+ is
    the Moore-Penrose pseudo-inverse of Y, easily computed from
    the SVD of Y.

    Args:
        X (np.2darray): each column is a datapoint. size nxk, where n is the
            dimension of the initial space and k is the number of points.
        Y (np.2darray): each column is a projection. size mxk, where m is the
            dimension of the projection space and k is the number of points.
        y0 (np.2darray): possible offset column vector added to points in Y
            after the projection.

    Returns:
        Q (np.2darray): a nxm inverse projection matrix that minimizes the
            squared error.
    """
    # Remove offsets
    Y2 = Y - y0 if (y0 is not None) else Y

    # Perform SVD
    # If matrices are too large, use full_matrices=False
    # It is always faster anyways. 
    full_mats = bool(max(*Y2.shape) < 4096)
    u, s, vh = np.linalg.svd(Y2, full_matrices=full_mats)
    # u is mxm, s is mxk, vh is kxk if full_mats
    # u is mxq, s is qxq, vh is qxk if not full_mats
    # where q = min(m, k)

    # Compute pseudo-inverse
    k, m = Y2.shape[1], Y2.shape[0]
    inverter = np.vectorize(lambda x: 1/x if x != 0 else 0)
    q = min(k, m)
    if full_mats:
        splus = np.zeros([k, m]) # kxm matrix
    else:
        splus = np.zeros([q, q])
    splus[:q, :q] = np.diagflat(inverter(s))
    
    # This dot product works whether full_mats or not
    Yplus = (vh.conj().T).dot(splus).dot(u.conj().T)

    # Compute and return Q
    return np.dot(X, Yplus)

# Custom class for linear regression using the pseudo-inverse and lsq_inverse function defined above
class PInverseRegression(BaseEstimator):
    """ Simple class for linear regression using my own method. Probably not as efficient
        as the sklearn one, but at least we know exactly what it is doing.

    Attributes:
        Q (np.ndarray): matrix of coefficients taking latent space points
            (each column is for one latent dimension) and back-projecting
            them to concentration space (one row for each output dimension)
        y0 (np.ndarray): offsets to subtract from Y. Zeros by default.
            Column vector.
            X = Q(Y - y0) where each column of X and Y
            is one point in their respective spaces
            (i.e. transpose of usual sklearn's order)

    Methods:
        fit: find the best matrix coefficients
        predict: project the input points back to X's space
        score: predict, and then compute the R^2 score
    """
    def __init__(self, y0=None):
        super().__init__()
        self.y0 = y0
        self.Q = None

    def fit(self, y, x):
        """ Assume each row of y (latent space) and x (reconstructed space) is a point.
        So, need to transpose x and y to respect lsq_inverse's working.
        """
        self.Q = lsq_inverse(x.T, y.T, self.y0)
        return self

    def predict(self, y):
        """ Assume each row of y is a point, so need to transpose to match the fact
        that Q's columns are for latent space dimensions (inputs to the regressor).
        """
        if self.Q is None:
            raise AttributeError("Regression not fitted yet, can't predict")

        x = np.dot(self.Q, y.T)  # Right now each column in x is a point

        if self.y0 is not None:
            x += self.y0

        # Transpose so again each row is a point
        return x.T

    def score(self, y, x):
        # Predicted output
        pred = self.predict(y)
        # Using sum twice to make sure we sum everything even with dataframes
        r2 = 1 - ((x - pred)**2).sum().sum()/((x - x.mean(axis=0))**2).sum().sum()

        return r2


## Functions and classes for regression with quadratic terms
# Short function to add quadratic features for an arbitrary number of input dimensions
# Mind that n input features give n + n(n-1)/2 extra quadratic features
# Short function to add quadratic features for an arbitrary number of input dimensions
# Mind that n input features give n + n(n-1)/2 extra quadratic features
def add_quadratic(y, n=None, which=[]):
    """ Short function to add quadratic terms to the input features. Each column is a feature.
    Order will be [squared terms], then [n1*n2, n1*n2, ..., n1*nn, n2*n3, ... nn-1 * nn].
    Updates the self.yq_ attribute.
    Args:
        y (2d array or DataFrame): each column is a feature
        n (int): number of columns of y for which to add quadratic terms.
            The quadratic features are added after the last column of y which
            may not be part of squared features.
            Mostly there for backwards comptability; superseded by which
        which (list): index of the columns to square. If both n and which
            are left to default, all terms are squared.
    Return:
        yq (2darray): y with quadratic terms appended to the right as extra columns.
    """
    yq = np.asarray(y)  # Copy and make it a pure array
    # Treat a few different input cases; check for errors later
    if yq.ndim == 1:  # y is a 1d array; make y2 a 2d array
        yq = yq[:, None]
        n = 1
        which = [0]
        print("1D array found")
    elif n is None and len(which) == 0:
        n = yq.shape[1]
        which = list(range(n))
        print("Squaring all terms by default")
    elif n is None:  # len(which) > 0
        n = len(which)
    elif len(which) == 0:  # n is not None
        n = int(n)
        which = list(range(n))
    else:
        print("List parameter which supersedes n_to_square")
        n = len(which)

    # Check for a few potential errors
    if n > yq.shape[1]:
        n = yq.shape[1]
        which = list(range(n))
        print("Specified number of features to square was too large")
    elif max(which) >= yq.shape[1]:
        raise IndexError("Specified feature column outside of range")

    # Squared terms
    y2 = yq[:, which]**2

    # Mixed terms, for the first n columns
    if n == 1:  # no mixed terms of course
        yq = np.concatenate([yq, y2], axis=1)
    else:
        ymix = []
        for i in range(n):
            for j in range(i+1, n):  # i < j to ensure mixed terms
                ymix.append(yq[:, which[i]] * yq[:, which[j]])
        ymix = np.asarray(ymix).T
        # Add the new columns with concatenation along axis 1
        yq = np.concatenate([yq, y2, ymix], axis=1)
    return yq


## Custom class for quadratic regression
class QuadraticRegression(BaseEstimator):
    """ Simple class for linear regression adding quadratic terms internally.
    We do not need to make those features explicit, because if n1 and n2 are
    scaled between 0 and 1, then terms n1^2, n2^2, and n1*n2 all remain
    in [0, 1].


    Attributes:
        Q (np.ndarray): matrix of coefficients taking latent space points
            (each column is for one latent dimension and quadratic terms)
            and back-projecting them to concentration space
            (one row for each output dimension)
            If there are n input features, there are 2n + n(n-1)/2 columns
            (linear plus quadratic features)
        y0 (np.ndarray): offsets to subtract from Y. Zeros by default.
            Column vector. Quadratic terms are computed once y0 is subtracted.
            Then X = QY', where each column of X and Y
            is one point in their respective spaces
            (i.e. transpose of usual sklearn's order)
        y_ (np.ndarray): last input points used, with quadratic features added
            as extra columns. y_ is just an array, if a df is given, it is
             stripped from its indexing.
        n_to_square (int): The first n_to_square columns will be used for
            quadratic terms. Useful to exclude some columns from the quadratic
            terms. If None, all features are squared.
            Mostly there for backwards compatibility
        which_to_square (list): column index of the features to square.
            Supersedes n_to_square if both are given.


    Methods:
        fit: find the best matrix coefficients
        predict: project the input points back to X's space
        score: predict, and then compute the R^2 score
    """
    def __init__(self, y0=None, n_to_square=None, which_to_square=[]):
        super().__init__()
        self.y0 = y0
        self.Q = None
        self.which_to_square = which_to_square
        if len(which_to_square) > 0:
            self.n_to_square = None
        else:
            self.n_to_square = n_to_square
            print("n_to_square =", self.n_to_square)
        self.yq_ = None  # Latest input points, one point per row, with quadratic features computed
        # after subtraction of y0

    def fit(self, y, x):
        """ Assume each row of y (latent space) and x (reconstructed space) is a point.
        So, need to transpose x and y to respect lsq_inverse's working.
        """
        print("Fitting")
        self.yq_ = add_quadratic(y, n=self.n_to_square,
                                    which=self.which_to_square)
        self.Q = lsq_inverse(x.T, self.yq_.T, self.y0)
        return self

    def predict(self, y):
        """ Assume each row of y is a point, so need to transpose to match the fact
        that Q's columns are for latent space dimensions (inputs to the regressor).
        """
        if self.Q is None:
            raise AttributeError("Regression not fitted yet, can't predict")
        if self.y0 is not None:
            x += self.y0

        # Add quadratic terms after shifting by y0
        self.yq_ = add_quadratic(y, n=self.n_to_square,
                                    which=self.which_to_square)

        # Predict
        x = np.dot(self.Q, self.yq_.T)  # Right now each column in x is a point

        # Transpose so again each row is a point
        return x.T

    def score(self, y, x):
        # Predicted output
        pred = self.predict(y)
        # Using sum twice to make sure we sum everything even with dataframes
        r2 = 1 - ((x - pred)**2).sum().sum()/((x - x.mean(axis=0))**2).sum().sum()

        return r2


## Main function for training a reconstruction algorithm of the selected method
def train_reconstruction(df_in, df_out, feature="integral", method="linear", model_args=dict(), do_scale_out=False):
    """ Use the data points in df_in (latent space) and the corresponding points
    in df_out (cytokine space) to train a regression algorithm to predict df_out given df_in.
    In other words, we optimize a function f(Y)=X, where Y is the latent space
    and X the cytokine space.

    Args:
        df_in (pd.DataFrame): index gives the peptide conditions, like
            (Data, Peptide, Concentration, TCellNumber, etc)
            and columns are Feature (integral, concentration, derivative)
            and Node (Node 1, Node 2). Each row is a point in Y.
        df_out (pd.DataFrame): same index as df_in, but cytokines instead of nodes.
            Each row is a point in Y
        feature (str): which feature type to use to find Q.
            Either "integral" (default), "concentration", or "derivative" (not recommended)
        method (str): "linear", "quadratic", "mlp", "mixed_quad",
            or "quad_mlp" (not implemented yet)
            Also supports "sklin" for comparison with my linear regression.
            LinearRegression.coef_ should be exactly like our PInverseRegression.Q,
            shape (n_target, n_features) where n_target is the number of cytokines
            For "mixed_quad", feature is disregarded for the inputs; all
            columns in df_in are kept.
        model_args (dict): keyword parameters to initialize the Regression model
        do_scale_out (bool): if True, the output is rescaled before fitting the regression,
            and the predictions are scaled back. If not, the fitting is done on raw outputs.

    Returns:
        pipeline (sklearn.pipeline.Pipeline): the full pipeline that transforms the input and output,
            and then predicts the reconstruction (automatic inversion of the output's transform).
    """
    y = df_in[feature] # Points in latent space, rows are node 1 and node 2
    x = df_out[feature]  # The points in original space, each row is a dimension

    # Scale the output too; it does not change anything for integrals since they already are between 0 and 1
    # It does help for concentrations however.
    if do_scale_out:
        outscaler = MinMaxScaler().fit(x)
    else:
        outscaler = FunctionTransformer()  # identity

    # Without a hidden layer, mlp would just be a linear regression, but worse.
    if method == "mlp":
        reg = MLPRegressor(activation="logistic", hidden_layer_sizes=(4,), max_iter=5000,
                    solver="lbfgs", random_state=92, alpha=0.0001, **model_args)

        # Parameters to grid search, this will be in a transformtarget in a pipeline, hence the __
        # Other parameters not used by lbfgs, which works fine for small datasets
        p_grid = dict(mlp__regressor__alpha=np.logspace(-5, -2, 4),
                      mlp__regressor__activation=["logistic", "tanh"])    # Relu creates weird bumps

        # The input scaler will be fitted when the full pipeline is fitted
        inscaler = MinMaxScaler()

    elif method == "linear":
        reg = PInverseRegression(y0=None, **model_args)
        p_grid = None
        # It's not good to rescale the projections without fitting intercepts too, so
        # use an identity function. Useless step, but it's for the sake of uniformity
        inscaler = FunctionTransformer()

    elif method == "sklin":
        reg = LinearRegression(fit_intercept=True)
        # This is really equivalent to my least-squares solution using the pseudo-inverse
        # but the MinMaxScalers below help a little bit to reduce the error.
        # With fit_intercept = False, my method gives EXACTLY the same result
        p_grid = None
        inscaler = FunctionTransformer()

    elif method == "quadratic":
        reg = QuadraticRegression(y0=None, **model_args)
        p_grid = None
        inscaler = FunctionTransformer()

    elif method == "mixed_quad":
        # Disregard the feature argument for inputs; keep all inputted columns
        # Assume which_to_square in model_args is a list of ints matching the
        # order of the columns inputted in df_in (user's responsibility to
        # check that, as if they were using .iloc).
        y = df_in.values
        reg = QuadraticRegression(y0=None, **model_args)
        p_grid = None
        inscaler = FunctionTransformer()

    elif method == "quad_mlp":
        # The input scaler will be fitted when the full pipeline is fitted
        inscaler = MinMaxScaler()
        raise NotImplementedError("Method 'quad_mlp' not tested yet")

    else:
        raise NotImplementedError(str(method))

    # Add the transform of the outputs in a TTR, so it is automatically inverted when predicting
    fullreg = TransformedTargetRegressor(regressor=reg, transformer=outscaler)

    # Define a pipeline with a MinMaxScaler (to rescale concentrations) and the MLPRegressor.
    # The pipeline makes sure that only the part of the training data used for fitting during
    # cross-validation is used to set the scale. See 3.1.1 in https://scikit-learn.org/stable/modules/cross_validation.html
    # We can even GridSearch over setting steps on or off, changing the steps, etc.
    pipe = Pipeline([('scaler', inscaler), (method, fullreg)])

    # Perform the gridsearch and use the refitted pipeline
    if p_grid is not None:
        print("Starting the grid parameter search with 5-fold CV: will take some time.")
        grid = GridSearchCV(pipe, p_grid, n_jobs=-1, refit=True).fit(y, x)
        print("Best parameters found:", grid.best_params_)
        pipe = grid.best_estimator_

        # Equivalent to:
        # grid = GridSearchCV(pipe, p_grid, n_jobs=-1, refit=False).fit(y, x)
        # pipe.set_params(**grid.best_params_)
        # pipe.fit(y, x)
    else:
        pipe.fit(y, x)
    score = pipe.score(y, x)
    return pipe, score


### Class for KDE fitting on parameter space
class ScalerKernelDensity(BaseEstimator):
    """ Takes the same arguments, has the same methods, etc. as a KernelDensity estimator,
    but with an additional mandatory positional argument scaler. This is a sklearn transform,
    pre-fitted to the desired range/scale, that is applied before fitting any data on the KDE,
    or inverted after sampling from the KDE.

    The number of columns in the sample points inputted to some methods must match the
    number of features for which the scaler was fit.

    fit(X[, y, sample_weight])         Fit the Kernel Density model on the data.
    get_params([deep])                 Get parameters for this estimator.
    sample([n_samples, random_state])  Generate random samples from the model.
    score(X[, y])                      Compute the total log probability density under the model.
    score_samples(X)                   Evaluate the log density model on the data.
    set_params(**params)               Set the parameters of this estimator.
    """
    def __init__(self, *, scaler=None, bandwidth=1.0, algorithm='auto',
                 kernel='gaussian', metric='euclidean', atol=0, rtol=0,
                 breadth_first=True, leaf_size=40, metric_params=None):
        if scaler is None:
            scaler = MinMaxScaler()
        self.scaler = scaler
        self.kde = KernelDensity(bandwidth=bandwidth, algorithm=algorithm,
                        kernel=kernel, metric=metric, atol=atol, rtol=rtol,
                        breadth_first=breadth_first, leaf_size=leaf_size,
                        metric_params=metric_params)

    def fit(self, X, y=None, sample_weight=None):
        # Will raise an error if the dimensions don't match
        Xtransform = self.scaler.transform(X, copy=True)
        self.kde.fit(Xtransform, y=y, sample_weight=sample_weight)
        return self

    def sample(self, n_samples=1, random_state=None):
        samples = self.kde.sample(n_samples, random_state)
        return self.scaler.inverse_transform(samples, copy=True)

    def score(self, X, y=None):
        Xtransform = self.scaler.transform(X, copy=True)
        return self.kde.score(Xtransform, y=y)

    def score_samples(self, X):
        Xtransform = self.scaler.transform(X, copy=True)
        return self.kde.score_samples(Xtransform)

    def get_params(self, deep=True):
        parms = self.kde.get_params(deep=deep)
        parms.update({"scaler":self.scaler})
        return parms

    def set_params(self, **params):
        return self.kde.set_params(**params)

    def gridsearch_fit_kde(self, X, param_grid=dict(), n_jobs=-1, refit=True, cv=2, **kwargs):
        """ Transform X once, then perform a GridSearchCV on the KDE of this custom class. """
        Xtransform = self.scaler.transform(X, copy=True)
        self.kde = GridSearchCV(self.kde, param_grid=param_grid, n_jobs=n_jobs, refit=refit, cv=cv,
                                **kwargs).fit(Xtransform).best_estimator_
        return self


# Companion function to fit a ScalerKernelDensity on each group of a dataset
def fit_param_distrib_kdes(df_p, df_v2v1, group_lvls=["Peptide"]):
    """Given a dataframe of parameter values, and a list of level(s) by which to group
    the parameter points, fit a KDE on each group, for all columns in the df.
    So please input only the parameter columns, not the parameter variance ones.
    Also fit a KDE on the distribution of v2/v1 ratios.

    Args:
        df_p (pd.DataFrame): each column is a parameter that the KDE will include.
        df_v2v1 (pd.Series): a series of v2/v1 ratios for some datasets.
        group_lvls (list): df_p levels by which to group the parameters.
            One KDE is fitted per group.
    Returns:
        kdes (dict): dictionary of ScalerKernelDensity objects,
            keys are the group keys.
        v2v1_kde (sklearn.neighbors.KernelDensity): KernelDensity estimate
            of the distribution of v2/v1 ratios.
    """
    # Fit a StandardScaler on the data
    sclr = StandardScaler()
    sclr.fit(df_p.values)

    # For each peptide, fit a KDE with the same scaler.
    kdes = dict()
    for key, pts in df_p.groupby(group_lvls):
        print(key, pts.shape)
        if pts.shape[0] < 2: continue

        kdes[key] = ScalerKernelDensity(scaler=sclr, kernel='gaussian').gridsearch_fit_kde(
            pts, param_grid={"bandwidth":np.arange(0.05, 0.3, 0.05)},
            n_jobs=-1, refit=True, cv=2)
        #kdes[key] = ScalerKernelDensity(scaler=sclr, kernel='gaussian').fit(pts)

    # v2/v1 ratios
    v2v1_kde = GridSearchCV(KernelDensity(kernel='gaussian'),
                param_grid={"bandwidth":np.arange(0.05, 1., 0.05)},
                n_jobs=-1, refit=True, cv=2
            ).fit(df_v2v1.values.reshape(-1, 1)).best_estimator_

    return kdes, v2v1_kde


# Companion function to sample from different KDEs, build a parameter df from it
# Sample new parameters from KDEs
# TODO: generalize a bit to select index levels better.
def sample_from_kdes(kde_dict, param_names, model_fit, nbs_replicates, seed=None,
    time_scale=20, duration=72):
    """ Function to create a DataFrame of synthetic sampled parameters. """
    # Create a RandomState with the seed. Just giving the same int seed to KDE
    # for each peptide would be very bad: the same random numbers would be drawn for
    # all peptides, so each replicate would have all its peptides heavily correlated.
    rng = np.random.RandomState(seed)
    if len(nbs_replicates) != len(kde_dict):
        raise ValueError("Must have one number of replicates for each kde")

    # Prepare to clip the sampled values!
    bounds_dict = {
        'Constant velocity':[(0, 0, 0, 0), (5, 5, np.pi, 5)],
        'Constant force': [(0, 0, 0, 0), (5, 5, np.pi, 5)],
        'Sigmoid':[(0, 0, -2*np.pi/3, 0, time_scale/50),
                    (5, (duration + 20)/time_scale, np.pi/3, 1, time_scale/2)],
        'Sigmoid_freealpha':[(0, 0, -2*np.pi/3, 0, time_scale/50, time_scale/50),
                            (5, (duration + 20)/time_scale, np.pi/3, 1,
                            time_scale/2, time_scale/2)]
    }
    clipminmaxs = np.asarray(bounds_dict[model_fit])
    # Prepare the DataFrame
    pep_keys = []
    replicate_keys = []
    for pep in kde_dict.keys():
        pep_keys += [pep]*nbs_replicates[pep]
        replicate_keys += list(map(str, range(1, nbs_replicates[pep]+1)))
    index = pd.MultiIndex.from_arrays([pep_keys, replicate_keys], names=["Peptide", "Replicate"])

    df_p_synth = pd.DataFrame(np.zeros([len(index), len(param_names)]), index=index, columns=param_names)
    df_p_synth.columns.name = "Parameter"
    # Sample for each peptide
    for pep in kde_dict.keys():
        reps = df_p_synth.loc[pep].shape[0]
        # Clip results
        df_p_synth.loc[pep] = np.clip(kde_dict[pep].sample(reps, random_state=rng),
                                           a_min=clipminmaxs[0], a_max=clipminmaxs[1])


    return df_p_synth


#
def compute_latent_curves(df_p, ser_v2v1, tanh_norm, times,
    model="sigmoid_freealpha", tsc=20.):
    """
    Function to compute the latent space curves for a df of model parameters.

    Args:
        df_p (pd.DataFrame): model parameters for various conditions (index)
        ser_v2v1 (pd.Series): the v2/v1 ratio for each row in df_p
        tanh_norm (pd.Series): the normalization factor for the integrals
            of Node 1 and Node 2 integrals in the tanh for reconstruction.
        times (np.ndarray): time points at which to compute curves
        model (string): name of the model, must be in {"Constant velocity",
            "Constant force", "Sigmoid", "Sigmoid_freealpha"}
        tsc (float): time scale, by which to rescale time (hours),
            must be the same used to fit the model parameters

    Returns:
        df_latent_synth (pd.DataFrame): DataFrame of latent space curves.
    """
    # Select appropriate functions.
    int_func_dict = {'Constant velocity':ballistic_v0,
                 'Constant force': ballistic_F,
                 'Sigmoid':ballistic_sigmoid,
                 'Sigmoid_freealpha': ballistic_sigmoid_freealpha}
    conc_func_dict = {"Sigmoid_freealpha": sigmoid_conc_full_freealpha}

    if model != "Sigmoid_freealpha":
        raise NotImplementedError("Currently, only the Sigmoid model with"
                    + "with free alpha is accepted for cytokine generation")

    conc_model = conc_func_dict.get(model)
    int_model = int_func_dict.get(model)
    blocks = dict()
    for key in df_p.index:
        params = list(df_p.loc[key])
        v2v1 = ser_v2v1[key]  # v2/v1 ratio

        # Compute the new values, rescale, and compare
        conc_values = conc_model(times/tsc, params, v2v1_ratio=v2v1)
        conc_values /= tsc

        # Also the integrals
        integ_values = int_model(times/tsc, *params, v2v1_ratio=v2v1)
        integ_values = np.tanh(integ_values / tanh_norm.values.T[:, np.newaxis])

        # replace
        blocks[key] = pd.DataFrame(np.concatenate([conc_values.T, integ_values.T], axis=1),
                            index=times, columns=pd.MultiIndex.from_product(
                            [["concentration", "tanh integral"], ["Node 1", "Node 2"]],
                            names=["Feature", "Node"]))

    # At the end, concatenate all blocks
    return pd.concat(blocks, axis=0, names=df_p.index.names+["Time"])

### Plotting function to compare reconstruction and data,
## or fit and data (latent space)
# Function to plot a comparison of the reconstruction and the original data, for a given feature
def plot_recon_true(df_full, df_r_full, feature="integral", toplevel="Data",
    sharey=True, do_legend=True, palette=palette_backup, pept=peptides_backup):
    """
    Args:
        df_full (pd.DataFrame): the dataframe for the experimental data
        df_r (pd.DataFrame): the reconstructed data
        feature (str): the feature to compare ("integral", "concentration", "derivative")
        toplevel (str): the first index level, one plot per entry
        sharey (bool): whether or not the y axis on each row should be shared
            True by default, allows to see if somne cytokines weigh less in the reconstruction.
        do_legend (bool): if True, add a legend for line styles
        palette (list): list of colors, at least as long as pept
        pept (list): list of peptides
    """
    # Slice for the desired feature
    df = df_full.xs(feature, level="Feature", axis=1, drop_level=True)
    df_r = df_r_full.xs(feature, level="Feature", axis=1, drop_level=True)

    # Plot the result
    # Rows are for peptides, columns for cytokines
    # One panel per dataset
    figlist = {}
    for xp in df.index.get_level_values(toplevel).unique():
        # Extract labels
        cols = df.loc[xp].index.get_level_values("Peptide").unique()
        cols = [p for p in pept if p in cols]  # Use the right order
        try:
            rows = df.loc[xp].columns.get_level_values("Cytokine").unique()
        except KeyError:
            rows = df.loc[xp].columns.get_level_values("Node").unique()
            print("Reconstructing latent space")
        concs_num = sort_SI_column(df.index.get_level_values("Concentration").unique(), "M")
        concs = np.asarray(df.index.get_level_values("Concentration").unique())[np.argsort(concs_num)]
        colors = {cols[i]:palette[i] for i in range(len(cols))}
        sizes = {concs[i]:1 + 0.3*i for i in range(len(concs))}

        # Make sure axes is a 2D array
        fig, axes = plt.subplots(len(rows), len(cols), sharex=True, sharey=sharey)
        if len(rows) == 1:
            axes = np.asarray([axes])
        elif len(cols) == 1:
            axes = axes[:, None]
        fig.set_size_inches(3*len(cols), 3*len(rows))
        times = df.loc[xp].index.get_level_values("Time").unique()
        times = [float(t) for t in times]
        for i, cyt in enumerate(rows):
            for j, pep in enumerate(cols):
                for k in concs:
                    try:
                        li1, = axes[i, j].plot(times, df.loc[(xp, pep, k), cyt],
                                    color=colors[pep], lw=sizes[k], ls="--")
                        li2, = axes[i, j].plot(times, df_r.loc[(xp, pep, k), cyt],
                                    color=colors[pep], lw=sizes[k], ls="-")
                    except KeyError:  # This combination dos not exist
                        continue
                # Some labeling
                if j == len(cols) - 1 and do_legend:
                    axes[i, j].legend([li1, li2], ["Original", "Reconstructed"],
                                      loc='upper left', bbox_to_anchor=(1, 1))
                if j == 0:
                    axes[i, j].set_ylabel(cyt)
                if i == len(rows) - 1:
                    axes[i, j].set_xlabel("Time")
                if i == 0:
                    axes[i, j].set_title(pep)
        # Save the figure afterwards
        figlist[xp] = fig
    return figlist


## Scoring/performance evaluation functions
# Function to compute some performance metrics
def performance_recon(df_full, df_r_full, feature="integral", toplevel=None):
    """
    Args:
        df_full (pd.DataFrame): the dataframe for the experimental data
        df_r (pd.DataFrame): the reconstructed data
        feature (str): the feature to compare ("integral", "concentration", "derivative")
        toplevel (str): the first index level, one plot per entry
    Returns:
        res_per_point
        histogs
        bins
        r2
    """
    # Slice for the desired feature
    df = df_full.xs(feature, level="Feature", axis=1, drop_level=True)
    df_r = df_r_full.xs(feature, level="Feature", axis=1, drop_level=True)

    # Residual vectors
    residuals = df - df_r
    # Squared norm of each residual vector
    residuals_norms = (residuals**2).sum(axis=1)

    # If toplevel is specified, do this per entry
    # If not, only one pass in the loop.
    if toplevel is None:
        totalres = residuals_norms.sum()
        nb = residuals_norms.shape[0]
        # histogram of the residuals, per dimension
        ndims = len(residuals.columns)
        histogs = pd.Series([[]]*ndims, index=residuals.columns)
        bins = pd.Series([[]]*ndims, index=residuals.columns)
        for dim in residuals.columns:
            histogs[dim], bins[dim] = np.histogram(residuals[dim])
    else:
        # Total residuals
        totalres = residuals_norms.groupby(toplevel).sum()
        # Divided by number of points in each group
        nb = residuals_norms.groupby(toplevel).count()
        # histogram of the residuals, per dimension and per dataset
        ndims = len(residuals.columns)
        topindex = residuals.index.get_level_values(toplevel).unique()
        nind = len(topindex)
        histogs = pd.DataFrame([[[]]*ndims]*nind, columns=residuals.columns, index=topindex)
        bins = histogs.copy()
        for dim in residuals.columns:
            for key in topindex:
                histogs.at[key, dim], bins.at[key, dim] = np.histogram(residuals.loc[key, dim])

    # In both cases, res_per_point is defined the same way.
    res_per_point = np.divide(totalres, nb)

    # R2 coefficient, for comparison with sklearn
    r2 = 1 - residuals_norms.sum()/((df - df.mean(axis=0))**2).sum().sum()

    return res_per_point, histogs, bins, r2


def plot_histograms(df_points, df_bins, xlabel="Residuals"):
    try:
        cols = df_points.columns
    except AttributeError:  # just a Series
        df_points = pd.DataFrame(df_points, index=[0], columns=df_points.index)
        df_bins = pd.DataFrame(df_bins, index=[0], columns=df_bins.index)
        cols = df_points.columns

    rows = df_points.index
    fig, axes = plt.subplots(len(rows), len(cols), sharey=True)  # can share y, same nb of points in each component
    if len(rows) == 1:
        axes = axes.reshape(1, len(cols))  # Make sure it's 2D
    fig.set_size_inches(1.5*len(cols), 1.5*len(rows))  # small histograms indeed
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            axes[i, j].hist(df_bins.loc[r, c][:-1], bins=df_bins.loc[r, c], weights=df_points.loc[r, c])
    for ax in axes[-1, :]:
        ax.set_xlabel(xlabel)
    return fig, axes

## TODO: code some unit tests for the methods above
def tests():
    pass

if __name__ == "__main__":
    tests()
