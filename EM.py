import numpy as np
from conditional_expectation_methods import compute_ols_parameters, nadaraya_watson, cluster_conditional_expectation, knn_conditional_expectation, isotonic_conditional_expectation, knn_conditional_expectation_improved


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
    1D Euler–Maruyama coupling with various conditional‐mean estimators,
    now also returns mY_hist and mX_hist of shape (N+1, n).
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

    # allocate histories for the conditional expectations
    mY_hist = np.empty((N+1, n))
    mX_hist = np.empty((N+1, n))

    # initial step: compute E[Y|X] and E[X|Y] at t=0
    X_old, Y_old = X0, Y0
    # dispatch just once for initial
    if isinstance(cond_method, str):
        method = cond_method.lower()
    else:
        method = None

    if method == 'nadaraya':
        m_Y_given_X = nadaraya_watson(X_old, X_old, Y_old, h)
        m_X_given_Y = nadaraya_watson(Y_old, Y_old, X_old, h)
    elif method == 'ols':
        m_Y_given_X, _, _ = compute_ols_parameters(X_old, Y_old)
        m_X_given_Y, _, _ = compute_ols_parameters(Y_old, X_old)
    elif method == 'isotonic':
        m_Y_given_X = isotonic_conditional_expectation(X_old, Y_old, X_old)
        m_X_given_Y = isotonic_conditional_expectation(Y_old, X_old, Y_old)
    elif method == 'cluster':
        m_Y_given_X = cluster_conditional_expectation(X_old, Y_old, k)
        m_X_given_Y = cluster_conditional_expectation(Y_old, X_old, k)
    elif method == 'knn':
        m_Y_given_X = knn_conditional_expectation(X_old, Y_old, k, algorithm='kd_tree')
        m_X_given_Y = knn_conditional_expectation(Y_old, X_old, k, algorithm='kd_tree')
    elif method == 'knn_improved':
        m_Y_given_X = knn_conditional_expectation_improved(X_old, Y_old, k, algorithm='kd_tree')
        m_X_given_Y = knn_conditional_expectation_improved(Y_old, X_old, k, algorithm='kd_tree')
    else:  # callable or error
        m_Y_given_X = cond_method(X_old, Y_old)
        m_X_given_Y = cond_method(Y_old, X_old)

    mY_hist[0] = m_Y_given_X
    mX_hist[0] = m_X_given_Y

    W2_sq = None
    if compute_W2_sq is not None:
        W2_sq = compute_W2_sq(X0, Y0)

    # main EM loop
    for i in range(N):
        X_old = X_traj[i]
        Y_old = Y_traj[i]

        # compute conditional-mean estimators
        if method == 'nadaraya':
            m_Y_given_X = nadaraya_watson(X_old, X_old, Y_old, h)
            m_X_given_Y = nadaraya_watson(Y_old, Y_old, X_old, h)
        elif method == 'ols':
            m_Y_given_X, _, _ = compute_ols_parameters(X_old, Y_old)
            m_X_given_Y, _, _ = compute_ols_parameters(Y_old, X_old)
        elif method == 'isotonic':
            m_Y_given_X = isotonic_conditional_expectation(X_old, Y_old, X_old)
            m_X_given_Y = isotonic_conditional_expectation(Y_old, X_old, Y_old)
        elif method == 'cluster':
            m_Y_given_X = cluster_conditional_expectation(X_old, Y_old, k)
            m_X_given_Y = cluster_conditional_expectation(Y_old, X_old, k)
        elif method == 'knn':
            m_Y_given_X = knn_conditional_expectation(X_old, Y_old, k, algorithm='kd_tree')
            m_X_given_Y = knn_conditional_expectation(Y_old, X_old, k, algorithm='kd_tree')
        elif method == 'knn_improved':
            m_Y_given_X = knn_conditional_expectation_improved(X_old, Y_old, k, algorithm='kd_tree')
            m_X_given_Y = knn_conditional_expectation_improved(Y_old, X_old, k, algorithm='kd_tree')
        else:  # callable
            m_Y_given_X = cond_method(X_old, Y_old)
            m_X_given_Y = cond_method(Y_old, X_old)

        # record them
        mY_hist[i] = m_Y_given_X
        mX_hist[i] = m_X_given_Y

        # compute gradients
        gU = grad_U(X_old)
        gV = grad_V(Y_old)

        # noise increments
        dW = sqrt_2eps_dt * np.random.randn(n)
        dB = sqrt_2eps_dt * np.random.randn(n)

        # EM updates
        X_new = X_old + (Y_old - m_Y_given_X - epsilon * gU)*dt + dW
        Y_new = Y_old + (X_old - m_X_given_Y - epsilon * gV)*dt + dB

        X_traj[i+1] = X_new
        Y_traj[i+1] = Y_new
        errors[i+1] = np.mean((X_new - Y_new)**2)

    # for completeness, store at the final time the last computed m
    mY_hist[N] = m_Y_given_X
    mX_hist[N] = m_X_given_Y

    return errors, X_traj, Y_traj, W2_sq, mY_hist, mX_hist


