import numpy as np
from sklearn.linear_model import LinearRegression
import itertools

def compute_ols_parameters(X_samples, Y_samples):
    if X_samples.ndim == 1:
        X_samples = X_samples.reshape(-1, 1)
    d = X_samples.shape[1]
    if d == 1:
        A = np.column_stack((X_samples, np.power(X_samples, 2), np.ones(X_samples.shape[0])))
    else:
        A = X_samples
        A_squared = np.power(X_samples, 2)
        A = np.column_stack((A, A_squared))
        cross_terms = []
        for i, j in itertools.combinations(range(d), 2):
            cross_terms.append(X_samples[:, i] * X_samples[:, j])
        A = np.column_stack((A, np.array(cross_terms).T))
        A = np.column_stack((A, np.ones(X_samples.shape[0])))
    ols = LinearRegression(fit_intercept=False)
    ols.fit(A, Y_samples)
    fhat = A @ ols.coef_.T
    return fhat, ols.coef_, A

def compute_conditional_mean(X, mu_X, mu_Y, Sigma_XX, Sigma_XY):
    X_diff = X - mu_X  # Element-wise difference
    Sigma_XX_inv = np.linalg.inv(Sigma_XX)
    mu_cond = mu_Y + X_diff @ Sigma_XX_inv @ Sigma_XY.T  # Conditional mean
    return mu_cond

def compute_relative_L2_error(f_hat, mu_cond):
    
    # Compute L2-norm of the difference (f_hat - mu_cond)
    norm_fhat_minus_fbar = np.sqrt(np.mean((f_hat - mu_cond) ** 2))

    # Compute L2-norm of the true conditional mean mu_cond
    norm_fbar = np.sqrt(np.mean(mu_cond ** 2))

    # Compute relative L2 error
    relative_L2_error = norm_fhat_minus_fbar / norm_fbar
    
    return relative_L2_error

def _gaussian_kernel(z):
    """ϕ(z) = exp(−z²/2) / √(2π)   – the normal pdf."""
    return np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)


def nadaraya_watson(x_query, X_sample, Y_sample, h, eps=1e-16):
    """
    Nadaraya–Watson estimator with leave‐one‐out correction.

    Parameters
    ----------
    x_query   : (nq,) array – points where you want m̂(x)
    X_sample  : (n,)  array – predictor samples
    Y_sample  : (n,)  array – response samples
    h         : float      – bandwidth
    eps       : float      – small constant to avoid division by zero

    Returns
    -------
    m_hat     : (nq,) array – NW estimates m̂(x_query)
    """
    # compute (nq, n) matrix of scaled distances
    z = (x_query[:, None] - X_sample[None, :]) / h
    K = _gaussian_kernel(z)  # shape (nq, n)

    # leave‐one‐out: if evaluating at the training points themselves,
    # zero out the self‐weights on the diagonal
    nq, n = K.shape
    if nq == n and np.allclose(x_query, X_sample):
        np.fill_diagonal(K, 0.0)

    # normalize weights, add eps to avoid division by zero
    row_sums = K.sum(axis=1, keepdims=True) + eps
    W = K / row_sums

    return W @ Y_sample

from sklearn.cluster import KMeans

def cluster_conditional_expectation(X: np.ndarray, 
                                    Y: np.ndarray, 
                                    k: int) -> np.ndarray:
    """
    Estimate E[Y | X] using clustering + Monte Carlo (piecewise-constant).
    
    Parameters
    ----------
    X : array-like, shape (n,)
        Predictor samples.
    Y : array-like, shape (n,)
        Response samples.
    k : int
        Number of clusters for KMeans.
    
    Returns
    -------
    m_hat : np.ndarray, shape (n,)
        Estimated conditional expectation values at each X_i.
    """
    X = np.asarray(X).reshape(-1, 1)
    Y = np.asarray(Y)
    # 1. Cluster X into k clusters
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    labels = kmeans.labels_
    
    # 2. Compute cluster-wise means of Y
    m_hat = np.zeros_like(Y, dtype=float)
    for cluster_id in range(k):
        mask = (labels == cluster_id)
        if np.any(mask):
            m_hat[mask] = Y[mask].mean()
        else:
            m_hat[mask] = 0.0  # fallback if empty cluster
    
    return m_hat

from sklearn.neighbors import NearestNeighbors

def knn_conditional_expectation(X: np.ndarray,
                                    Y: np.ndarray,
                                    k: int,
                                    algorithm: str = 'kd_tree') -> np.ndarray:
    """
    Estimate E[Y | X] using k-nearest-neighbors with a KD-tree (piecewise-constant).

    Parameters
    ----------
    X : array-like, shape (n,)
        Predictor samples.
    Y : array-like, shape (n,)
        Response samples.
    k : int
        Number of neighbors for kNN.

    Returns
    -------
    m_hat : np.ndarray, shape (n,)
        Estimated conditional expectation values at each X_i.
    """
    # 1. Prepare data
    X = np.asarray(X).reshape(-1, 1)   # make into (n, 1)-shaped array
    Y = np.asarray(Y)

    # 2. Build KD-tree on the predictor values
    #    'algorithm="kd_tree"' ensures use of a KD-tree index under the hood
    knn = NearestNeighbors(n_neighbors=k, algorithm=algorithm).fit(X)

    # 3. For each X_i, find the indices of its k nearest neighbors
    #    kneighbors returns (distances, indices) arrays of shape (n, k)
    _, neighbor_idxs = knn.kneighbors(X, return_distance=True)

    # 4. Compute the local average of Y over those neighbors:
    #    \hat m(X_i) = (1/k) * sum_{j in N_k(i)} Y_j
    m_hat = np.mean(Y[neighbor_idxs], axis=1)

    return m_hat
