import numpy as np
from conditional_expectation_methods import compute_ols_parameters, nadaraya_watson, cluster_conditional_expectation, knn_conditional_expectation


def euler_maruyama_coupling(
    X0: np.ndarray,
    Y0: np.ndarray,
    epsilon: float,
    T: float,
    N: int,
    cond_method: str,
    h: float = None,
    k: int = None,
    grad_U=None,
    grad_V=None,
    compute_W2_sq=None,
    seed: int = None,
):
    """
    1D Euler–Maruyama coupling with various conditional‐mean estimators.

    Parameters
    ----------
    X0, Y0 : array, shape (n,)
        Initial samples from μ and ν.
    epsilon : float
    T : float
    N : int
    cond_method : {'ols','nadaraya','cluster','knn'} or callable
        Which conditional‐mean to use:
          - 'ols': ordinary least squares regression
          - 'nadaraya': Nadaraya–Watson kernel regression (needs h)
          - 'cluster': piecewise-constant via KMeans (needs k)
          - 'knn': piecewise-constant via k-nearest neighbors (needs k, optional algorithm)
          - or user‐supplied function(X, Y) -> m_Y|X
    h : float, optional
        Bandwidth for Nadaraya–Watson (required if cond_method='nadaraya').
    k : int, optional
        Number of clusters or neighbors (required if cond_method in {'cluster','knn'}).
    grad_U, grad_V : callables
        Functions grad_U(X_old), grad_V(Y_old).
    compute_W2_sq : callable or None
    seed : int or None

    Returns
    -------
    errors : ndarray, shape (N+1,)
    X_traj, Y_traj : ndarray, shape (N+1, n)
    W2_sq or None
    """
    if seed is not None:
        np.random.seed(seed)

    n = X0.shape[0]
    dt = T / N
    sqrt_2eps_dt = np.sqrt(2 * epsilon * dt)

    X_traj = np.empty((N+1, n))
    Y_traj = np.empty((N+1, n))
    X_traj[0], Y_traj[0] = X0.copy(), Y0.copy()

    errors = np.empty(N+1)
    errors[0] = np.mean((X0 - Y0)**2)

    W2_sq = None
    if compute_W2_sq is not None:
        W2_sq = compute_W2_sq(X0, Y0)

    for i in range(N):
        X_old = X_traj[i]
        Y_old = Y_traj[i]

        # select conditional mean estimator
        if isinstance(cond_method, str):
            method = cond_method.lower()
        else:
            method = None

        if method == 'nadaraya':
            if h is None:
                raise ValueError("Bandwidth h must be provided for Nadaraya–Watson")
            m_Y_given_X = nadaraya_watson(X_old, X_old, Y_old, h)
            m_X_given_Y = nadaraya_watson(Y_old, Y_old, X_old, h)

        elif method == 'ols':
            m_Y_given_X, _, _ = compute_ols_parameters(X_old, Y_old)
            m_X_given_Y, _, _ = compute_ols_parameters(Y_old, X_old)

        elif method == 'cluster':
            if k is None:
                raise ValueError("Number of clusters k must be provided for clustering")
            m_Y_given_X = cluster_conditional_expectation(X_old, Y_old, k)
            m_X_given_Y = cluster_conditional_expectation(Y_old, X_old, k)

        elif method == 'knn':
            if k is None:
                raise ValueError("Number of neighbors k must be provided for kNN")
            # default to KD-tree, but allow other algorithms
            m_Y_given_X = knn_conditional_expectation(X_old, Y_old, k, algorithm='kd_tree')
            m_X_given_Y = knn_conditional_expectation(Y_old, X_old, k, algorithm='kd_tree')

        elif callable(cond_method):
            m_Y_given_X = cond_method(X_old, Y_old)
            m_X_given_Y = cond_method(Y_old, X_old)

        else:
            raise ValueError(f"Unknown cond_method: {cond_method}")

        # gradients
        gU = grad_U(X_old)
        gV = grad_V(Y_old)

        # noise
        dW = sqrt_2eps_dt * np.random.randn(n)
        dB = sqrt_2eps_dt * np.random.randn(n)

        # EM update
        X_new = X_old + (Y_old - m_Y_given_X - epsilon * gU) * dt + dW
        Y_new = Y_old + (X_old - m_X_given_Y - epsilon * gV) * dt + dB

        X_traj[i+1] = X_new
        Y_traj[i+1] = Y_new
        errors[i+1] = np.mean((X_new - Y_new)**2)

    return errors, X_traj, Y_traj, W2_sq
