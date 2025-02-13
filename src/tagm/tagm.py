"""T-augmented Gaussian Mixture Model."""

# Author: Ulrich Dobramysl

import random
import warnings
import numpy as np
from numpy.typing import NDArray
from scipy.special import loggamma as lgΓ
from scipy.stats import dirichlet, invwishart, beta

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import label_binarize
from sklearn.cluster import KMeans
from sklearn.utils.validation import validate_data
from sklearn.utils.multiclass import unique_labels

all = ["TAGMMAP"]

_LG2PI = np.log(2 * np.pi)

def log_gaussian(
    x: NDArray[np.float64],
    µ: NDArray[np.float64],
    Σ: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the log of the multivariate Gaussian function

    Arguments:
     - x: Points at which to evaluate the function, array of shape (N, K)
     - μ: Location of the multivariate Gaussian, array of shape (K,)
     - Σ: Covariance of the multivariate Gaussian, array of shape (K, K)

    Returns:
     - Value of the multivariate Gaussian at the given points, array of shape (N,)
    """
    K = µ.shape[0]

    sign, log_D = np.linalg.slogdet(Σ)
    if sign == 0:
        raise ValueError("sigma is singular.")

    x_µ = x - µ[np.newaxis, :]

    Σi = np.linalg.inv(Σ)
    exponent = ((x_μ @ Σi)[:, np.newaxis, :] @ x_μ[:, :, np.newaxis])[:, 0, 0]

    return -0.5 * (K * _LG2PI + log_D + exponent)


def log_t(
    x: NDArray[np.float64],
    ν: float,
    µ: NDArray[np.float64],
    Σ: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the log of the multivariate T distribution PDF

    Arguments:
     - x: Points at which to evaluate the function, array of shape (N, K)
     - ν: Degrees of freedom of the T distribution, positive float
     - μ: Location of the multivariate Gaussian, array of shape (K,)
     - Σ: Covariance of the multivariate Gaussian, array of shape (K, K)

    Returns:
     - Value of the multivariate Gaussian at the given points, array of shape (N,)
    """
    p = μ.shape[0]

    sign, log_D = np.linalg.slogdet(Σ)
    if sign == 0:
        raise ValueError("sigma is singular.")
    
    x_µ = x - µ[np.newaxis, :]
    Σi = np.linalg.inv(Σ)

    return (
        lgΓ(0.5 * (ν + p))
        - (lgΓ(0.5 * ν) + 0.5 * p * np.log(ν * np.pi) + 0.5 * log_D)
        - 0.5
        * (ν + p)
        * np.log(
            1 + ((x_μ @ Σi)[:, np.newaxis, :] @ x_μ[:, :, np.newaxis])[:, 0, 0] / ν
        )
    )

class _TAGMFitData:
    # Fitting data
    X: NDArray[np.float64]
    unknowns: NDArray[np.float64]
    y: NDArray[np.float64]

    # Dimensions
    N: int
    K: int
    D: int
    weights: NDArray[np.int64]      # (K,)

    # Priors
    beta_0: NDArray[np.float64]     # (K,)
    beta_k: NDArray[np.float64]     # (K,)
    mu_0: NDArray[np.float64]       # (D,)
    lambda_0: float
    nu_0: float
    S_0: NDArray[np.float64]        # (D, D)
    S_k: NDArray[np.float64]        # (K, D, D)
    kappa: float
    M: NDArray[np.float64]          # (D,)
    V: NDArray[np.float64]          # (D, D)
    u: float
    v: float

    # Constants
    m_k: NDArray[np.float64]        # (K, D)

    # Posteriors
    pi_k: NDArray[np.float64]       # (K,)
    mu_k: NDArray[np.float64]       # (K, D)
    sigma_k: NDArray[np.float64]    # (K, D, D)
    epsilon: float

    # Working data
    ab: NDArray[np.float64]         # (K, N, 2)

    def __init__(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        beta_0: float | NDArray[np.float64],
        lambda_0: float,
        nu_0: float | None,
        kappa: float,
        u: float,
        v: float,
    ):
        self.X = X
        self.y = y
        self.unknowns = self.X[~self.y.any(axis=1), :]

        # Initialise dimensions and label weights
        self.K = y.shape[1]
        self.D = X.shape[1]
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                "Number of observations in X and y second dimension needs to be the same."
            )
        self.weights = y.sum(axis=0)

        # Priors
        ## Initialise mixture weights Dirichlet priors
        if isinstance(beta_0, int) or isinstance(beta_0, float):
            self.beta_0 = beta_0 * np.ones(self.K)
        else:
            self.beta_0 = beta_0
        self.beta_k = self.beta_0 + self.weights

        ## Initialise Normal priors
        markerData = self.X[y.any(axis=1), :]
        self.N = markerData.shape[0]    # I'm pretty sure this is wrong, but pRoloc has that. I think it should be X.shape[0] instead.
        self.mu_0 = markerData.mean(axis=0)
        self.lambda_0 = lambda_0

        ## Initialise Normal Inverse Wishart priors
        if nu_0 is None:
            self.nu_0 = self.D + 2
        else:
            self.nu_0 = nu_0
        self.S_0 = np.diag(((markerData - markerData.mean())**2).sum(axis=0) / self.N) / (self.K ** (1 / self.D))

        ## Initialise multivariate T-distribution priors
        self.kappa = kappa
        self.M = X.mean(axis=0)
        self.V = np.cov(X, rowvar=False) / 2
        eigenvalues = np.linalg.eigvals(self.V)
        if eigenvalues.min() < np.finfo(float).eps:
            self.V += 1e-6 * np.identity(self.V.shape[0])
            warnings.warn(
                "Co-linearity detected. Added a small multiple of the identity to the covariance."
            )

        ## Initialise outlier priors
        self.u = u
        self.v = v


        # Initialise mixture weights
        self.pi_k = self.weights / self.weights.sum()

        # Initialise component mean and covariance

        ## Component mean
        self.x_k = np.empty((self.K, self.D))
        for k in range(self.K):
            self.x_k[k] = self.X[self.y[:, k], :].mean(axis=0)
        self.m_k = (
            self.weights[:, np.newaxis] * self.x_k
            + (self.lambda_0 * self.mu_0)[np.newaxis, :]
        ) / (self.lambda_0 + self.weights)[:, np.newaxis]
        self.mu_k = self.m_k.copy()

        ## Component covariance
        lambda_k = self.lambda_0 + self.weights
        self.S_k = np.empty((self.K, self.D, self.D))
        for k in range(self.K):
            XX: NDArray[np.float64] = X[y[:, k], :]
            self.S_k[k] = (
                self.S_0
                + (XX.T @ XX)
                + (self.lambda_0 * (self.mu_0[:, np.newaxis] * self.mu_0[np.newaxis, :]))
                - lambda_k[k] * (self.mu_k[k, :, np.newaxis] * self.mu_k[k, np.newaxis, :])
            )
        nu_k = self.nu_0 + self.weights
        self.sigma_k = self.S_k / (nu_k[:, np.newaxis, np.newaxis] + self.D + 1)

        # Initialise outlier component
        self.epsilon = (self.u - 1) / (self.u + self.v - 2)

        # Initialise data array
        self.ab = np.empty((self.K, self.unknowns.shape[0], 2))

    def log_f(self, k, x):
        return log_gaussian(x, self.mu_k[k], self.sigma_k[k])

    def log_g(self, x):
        return log_t(x, self.kappa, self.M, self.V)

    def log_p_phi1_non_normed(self, k, x):
        return np.log(self.pi_k[k]) + np.log(1 - self.epsilon) + self.log_f(k, x)

    def log_p_phi0_non_normed(self, k, x):
        return np.log(self.pi_k[k]) + np.log(self.epsilon) + self.log_g(x)



class TAGMMAP(ClassifierMixin, BaseEstimator):
    """T-augmented Gaussian Mixture classifier (maximum a posteriori version).
    
    The TAGM-MAP classification model introduced to assign localisation terms and organelles to high-throughput proteomics data (LOPIT-DC, HyperLOPIT, etc.). This enhances a regular Bayesian Gaussian Mixture with an outlier component distributed according to a multivariate T distribution [1, 2].

    Parameters
    ----------

    n_components : int, default=300
        The number of mixture components. This is only used when no training data is given in `fit` or `fit_predict`.

    max_iter : int, default=100
        The maximum number of iterations allowed. Shows a warning if this is reached without the model having converged.
    
    tol : float, default=1e-4
        The convergence criterion. Convergence is reached the difference between log likelihood of the current and the previous EM iteration is less than `tol`.

    weight_concentration_prior : float, default=1.0
        The concentration of each component on the Dirichlet weight distribution (similar to sklearn.mixture.BayesianGaussianMixture).

    mean_precision_prior : float, default=0.01
        The precision prior on the mean Gaussian distribution (similar to sklearn.mixture.BayesianGaussianMixture).

    degrees_of_freedom_prior : float or None, default=None
        The prior of the number of degrees of freedom on the covariance distributions (similar to sklearn.mixture.BayesianGaussianMixture). If None, it's set to `n_features+2`

    outlier_conjugate_shape_prior : tuple of floats, default=(2, 10)
        The parameters of the conjugate Beta distribution describing the outlier component.

    outlier_degrees_of_freedom : float, default=4
        The degrees of freedom of the multivariate T distribution describing the outlier component.

    Attributes:
    -----------

    weights_ : array-like of shape (n_components,)
        The weight of each mixture component.
    
    means_ : array-like of shape (n_components, n_features)
        The mean of each mixture component.
    
    covariances_ : array_like of shape (n_components, n_features, n_features)
        The covariance of each mixture component.
    
    converged_ : bool
        True if convergence was reached.
    
    n_iter_ : int
        Number of EM steps required to reach convergence (if convergence was reached).

    loglikelihood_ : array_like of shape (n_iter_,)
        Log-likelihood after each EM iteration.

    References
    ----------
    
    .. [1] `Crook OM, Mulvey CM, Kirk PDW, et al. A Bayesian mixture modelling approach for spatial proteomics. PLoS Computational Biology 14(11):e1006516 2018. <https://doi.org/10.1371/journal.pcbi.1006516>`
    
    .. [2] `Crook OM, Breckels LM, Lilley KS et al. A Bioconductor workflow for the Bayesian analysis of spatial proteomics. F1000Research 2019, 8:446 <https://doi.org/10.12688/f1000research.18636.1>` 

    """
    def __init__(
        self,
        n_components=300,
        max_iter=100,
        tol=1e-4,
        weight_concentration_prior=1.0,
        mean_precision_prior=0.01,
        degrees_of_freedom_prior=None,       # D + 2
        outlier_conjugate_shape_prior=(2, 10),
        outlier_degrees_of_freedom=4,
        random_state=None,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.weight_precision_prior = weight_concentration_prior
        self.mean_precision_prior = mean_precision_prior
        self.degrees_of_freedom_prior = degrees_of_freedom_prior
        self.outlier_degrees_of_freedom = outlier_degrees_of_freedom
        self.outlier_conjugate_shape_prior = outlier_conjugate_shape_prior
        self.random_state = random_state

    def _initialise_clusters(self, X):
        kmeans = KMeans(n_clusters=self.n_components, random_state=self.random_state)
        kmeans.fit(X)
        y = np.zeros((X.shape[0], self.n_components), dtype=bool)
        y[np.arange(y.shape[0]), kmeans.labels_] = True
        return y

    def _binarize(self, y, unknown_label):
        self.classes_ = np.array([unknown_label] + list(unique_labels(y[y != unknown_label])))
        return label_binarize(y, classes=self.classes_) > 0

    def _inverse_binarize(self, y):
        return self.classes_[y.argmax(axis=1)]

    def fit(self, X, y=None, unknown_label=-1):
        """Estimate model parameters with the EM algorithm.
        
        Fits the model by iterating the expectation-maximisation (EM) algorithm until either convergence is achieved or `max_iter` is reached.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, each row represents a single data point.

        y : None or array-like of shape (n_samples, n_outputs), default=None
            Training data, in the form of one row per data point given in `X`. Each column represents a different label. A data point for which the classification is known has at least one column that is `True`, but can have more than one.
            If None, the initial parameters are chosen by running a k-means clustering algorithm finding n_components clusters.

        unknown_label: str or int, default=-1
            Determines which label in `y` is treated as unknown. Common values are "unknown" or `-1`.

        Returns
        -------
        self : object
            The fitted TAGMMAP model.
        """

        X, y = validate_data(
            self,
            X,
            y,
            multi_output=True,
        )

        if y is None:
            y = self._initialise_clusters(X)
        y = self._binarize(y, unknown_label)

        self._data = _TAGMFitData(
            X,
            y[:, 1:],
            self.weight_precision_prior,
            self.mean_precision_prior,
            self.degrees_of_freedom_prior,
            self.outlier_degrees_of_freedom,
            self.outlier_conjugate_shape_prior[0],
            self.outlier_conjugate_shape_prior[1],
        )

        Q_prev: float = 0.0
        self.n_iter_ = 0
        loglikelihood: list[float] = []
        self.converged_ = False
        for _ in range(self.max_iter):
            self._estep()
            self._mstep()
            Q_new = self._eval_loglikelihood()
            self.n_iter_ += 1
            loglikelihood.append(Q_new)
            if abs(Q_new - Q_prev) < self.tol:
                self.converged_ = True
                break
            Q_prev = Q_new
        else:
            warnings.warn(
                "max_iter iterations reached and max likelihood convergence not achieved."
            )
        self.loglikelihood_ = np.array(loglikelihood)

        self.means_ = self._data.mu_k
        self.covariances_ = self._data.sigma_k
        self.weights_ = self._data.pi_k

        return self

    def _ab(self, X: NDArray[np.float64]):
        d = self._data
        lgab: NDArray[np.float64] = np.zeros((d.K, X.shape[0], 2))
        for k in range(d.K):
            lgab[k, :, 0] = d.log_p_phi1_non_normed(k, X)
            lgab[k, :, 1] = d.log_p_phi0_non_normed(k, X)

        # Correct for underflow
        lgab -= lgab.max(axis=2).max(axis=0)[np.newaxis, :, np.newaxis]
        ab: NDArray[np.float64] = np.exp(lgab)
        ab /= ab.sum(axis=0).sum(axis=1)[np.newaxis, :, np.newaxis]

        return ab

    def _estep(self):
        d = self._data
        d.ab[:] = self._ab(d.unknowns)

    def _mstep(self):
        d = self._data
        a = d.ab[:, :, 0].sum()
        b = d.ab[:, :, 1].sum()
        d.epsilon = (d.u + b - 1) / (a + b + d.u + d.v - 2)

        r = d.ab.sum(axis=2).sum(axis=1)
        d.pi_k = (
            (r + d.beta_k - 1) /
            (d.unknowns.shape[0] + d.beta_k.sum() - d.K)
        )

        xm_k: NDArray[np.float64] = (
            (d.ab[:, :, 0, np.newaxis] * d.unknowns[np.newaxis, :, :]).sum(axis=1) /
            d.ab[:, :, 0].sum(axis=1)[:, np.newaxis]
        )

        # The definitions of lambda_k and nu_k are different in pRoloc compared to the paper. Not sure why, but we're going with pRoloc here.
        a_k: NDArray[np.float64] = d.ab[:, :, 0].sum(axis=1)
        lambda_k = d.lambda_0 + d.weights
        nu_k = d.nu_0 + d.weights

        # Again, this uses lambda_k * d.m_k (the initial guess that isn't updated), but according to the paper this should be d.lambda_0 * d.mu_0 instead!
        # Let's copy pRoloc for now, but we might have to revisit.
        d.mu_k = (
            (a_k[:, np.newaxis] * xm_k + lambda_k[:, np.newaxis] * d.m_k) /
            (lambda_k + a_k)[:, np.newaxis]
        )

        # This /should/ be d.mu_0 according to the paer. The code in pRoloc uses mk.
        # Still, we want to do what pRoloc is doing, so stick with it.
        xkmuk = xm_k - d.m_k
        xixk = (
            (xkmuk[:, :, np.newaxis] * xkmuk[:, np.newaxis, :]) *
            ((lambda_k * a_k) / (lambda_k + a_k))[:, np.newaxis, np.newaxis]
        )
        for k in range(d.K):
            for i, x in enumerate(d.unknowns):
                diff: NDArray[np.float64] = d.unknowns[i] - xm_k[k]
                xixk[k, :, :] += d.ab[k, i, 0] * (diff[:, np.newaxis] * diff[np.newaxis, :])
        # pRoloc has S_k here instead of S_0 like in the paper.
        d.sigma_k = (d.S_k + xixk) / (nu_k + a_k + d.D + 2)[:, np.newaxis, np.newaxis]

    def _eval_loglikelihood(self):
        d = self._data

        Q: float = 0.0
        for k in range(d.K):
            Q += (
                (d.ab[k].sum(axis=1) * np.log(d.pi_k[k])).sum()
                + (d.ab[k, :, 0] * d.log_f(k, d.unknowns)).sum()
                + log_gaussian(d.mu_k[k], d.mu_0, d.sigma_k[k])[0]
            )
            Q += invwishart.logpdf(d.sigma_k[k], df=d.nu_0, scale=d.S_0)
        Q += (d.ab[:, :, 1].sum(axis=0) * d.log_g(d.unknowns)).sum()
        Q += d.ab[:, :, 0].sum() * np.log(1 - d.epsilon) + d.ab[:, :, 1].sum() * np.log(d.epsilon)
        Q += beta.logpdf(d.epsilon, d.u, d.v)
        Q += dirichlet.logpdf(d.pi_k, alpha=d.beta_0 / d.K)
        return Q

    def predict_proba(self, X: NDArray[np.float64]):
        """Evaluate the probability of each localisation of the data points in X.

        The first returned column indicates the probability that a data point is an outlier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, each row represents a single data point.

        Returns
        -------
        probability : array of shape (n_samples, n_outputs + 1)
            Probability of a data point belonging to each localisation component. The first column contains the probability of a data point being an outlier.
        """
        ab = self._ab(X)
        probability = np.empty((X.shape[0], self._data.K+1))
        probability[:, 1:] = ab[:, :, 0].T
        probability[:, 0] = ab[:, :, 1].sum(axis=0)
        return probability
    
    def predict(self, X: NDArray[np.float64]):
        """Predict the localisation for X.

        A label of `0` indicates that the data point was assigned to the outlier component.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, each row represents a single data point.

        Returns
        -------
        localisation : array of shape (n_samples,)
            Predicted localisation.
        """
        probabilities = self.predict_proba(X)
        # Ignore the outlier component for classification
        probabilities[:, 0] = 0
        return self._inverse_binarize(probabilities)
    
    def fit_predict(self, X, y, **kwargs):
        """Estimate model parameters and predict the localisation for X.
        
        Fits the model by iterating the expectation-maximisation (EM) algorithm until either convergence is achieved or `max_iter` is reached. After this, predicts the localisation of each data point by returning the localisation with the highest probability.

        A label of `0` indicates that the data point was assigned to the outlier component.

        Note, that this function is different from calling `fit(X, y)` followed by `predict`. This function won't assign a class for entries for which training data exists in `y`, instead returning the class given in `y`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, each row represents a single data point.

        y : array-like of shape (n_samples, n_outputs)
            Training data, in the form of one row per data point given in `X`. Each column represents a different label. A data point for which the classification is known has at least one column that is `True`, but can have more than one.

        Returns
        -------
        localisation : array of shape (n_samples,)
            Predicted localisation.
        """
        proba = self.fit_predict_proba(X, y, **kwargs)
        # Ignore the outlier component for classification
        proba[:, 0] = 0.0
        return self._inverse_binarize(proba)
    
    def fit_predict_proba(self, X, y, **kwargs):
        """Estimate model parameters and return the class probabilities for X.
        
        Fits the model by iterating the expectation-maximisation (EM) algorithm until either convergence is achieved or `max_iter` is reached. After this, returns the probability for each class.
        The first returned column indicates the probability that a data point is an outlier.

        Note, that this function is different from calling `fit(X, y)` followed by `predict_proba`. This function won't calculate any probabilities for entries for which training data exists in `y`, instead setting the probability for the corresponding class to one.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, each row represents a single data point.

        y : array-like of shape (n_samples, n_outputs)
            Training data, in the form of one row per data point given in `X`. Each column represents a different label. A data point for which the classification is known has at least one column that is `True`, but can have more than one.

        Returns
        -------
        localisation : array of shape (n_samples,)
            Predicted localisation.
        """
        self.fit(X, y, **kwargs)
        
        proba = np.zeros((self._data.X.shape[0], self._data.y.shape[1]+1))
        have_training = self._data.y.any(axis=1)
        proba[~have_training, :] = self.predict_proba(self._data.X[~have_training, :])
        proba[have_training, self._data.y[have_training].argmax(axis=1)+1] = 1.0
        return proba

