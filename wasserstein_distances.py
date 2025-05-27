import numpy as np
from scipy.linalg import sqrtm

def bures_wasserstein_distance_gaussian(m1, C1, m2, C2):
    """
    Compute the exact 2-Wasserstein (Bures–Wasserstein) distance between two Gaussian measures.

    Parameters
    ----------
    m1 : float or array-like, shape (d,)
        Mean of the first Gaussian.
    C1 : float or array-like, shape (d, d)
        Covariance (variance if float) of the first Gaussian.
    m2 : float or array-like, shape (d,)
        Mean of the second Gaussian.
    C2 : float or array-like, shape (d, d)
        Covariance (variance if float) of the second Gaussian.

    Returns
    -------
    W2 : float
        The 2-Wasserstein distance W₂(𝒩(m1, C1), 𝒩(m2, C2)).
    """
    # 1D case: variances are scalars
    if np.isscalar(C1) and np.isscalar(C2):
        return (m1 - m2)**2 + (C1 - C2)**2

    # Multi-dimensional case
    m1 = np.atleast_1d(m1)
    m2 = np.atleast_1d(m2)
    C1 = np.atleast_2d(C1)
    C2 = np.atleast_2d(C2)

    # Squared distance between means
    delta_sq = np.dot(m1 - m2, m1 - m2)

    # Bures term
    # sqrtC2 = C2^(1/2)
    sqrtC2 = sqrtm(C2)
    # Middle matrix: (sqrtC2 * C1 * sqrtC2)^(1/2)
    middle = sqrtm(sqrtC2 @ C1 @ sqrtC2)
    bures_sq = np.trace(C1 + C2 - 2 * middle)

    return np.sqrt(delta_sq + bures_sq)


def W2_empirical(x, y):
    xs = np.sort(x)
    ys = np.sort(y)
    return np.sqrt(np.mean((xs - ys)**2))


from scipy.linalg import sqrtm, det

def entropic_wasserstein_gaussian(m0, K0, m1, K1, epsilon):
    """
    Compute the entropic‐regularized 2‐Wasserstein cost
    OT_{d^2}^ε( N(m0, K0), N(m1, K1) )
    via Corollary 1 (a) in Mallasto et al. 2021.

    Parameters
    ----------
    m0 : array-like, shape (n,)
        Mean of the first Gaussian.
    K0 : array-like, shape (n,n)
        Covariance of the first Gaussian.
    m1 : array-like, shape (n,)
        Mean of the second Gaussian.
    K1 : array-like, shape (n,n)
        Covariance of the second Gaussian.
    epsilon : float
        Entropic regularization parameter ε > 0.

    Returns
    -------
    OT_eps : float
        The entropic‐regularized 2‐Wasserstein cost.
    """
    # Ensure arrays
    m0 = np.atleast_1d(m0)
    m1 = np.atleast_1d(m1)
    K0 = np.atleast_2d(K0)
    K1 = np.atleast_2d(K1)
    n = m0.shape[0]

    # 1) mean‐squared term
    delta2 = np.dot(m0 - m1, m0 - m1)

    # 2) trace terms
    tr0 = np.trace(K0)
    tr1 = np.trace(K1)

    # 3) build M^ε = I + ( I + (16/ε²) K0 K1 )^{1/2}
    A = np.eye(n) + (16.0 / epsilon**2) * (K0 @ K1)
    sqrtA = sqrtm(A)
    M_eps = np.eye(n) + sqrtA

    # 4) compute trace and log‐determinant of M^ε
    tr_M = np.trace(M_eps)
    sign, logdet_M = np.linalg.slogdet(M_eps)
    if sign <= 0:
        raise ValueError("M^ε must be positive‐definite; numerical issue encountered.")

    # 5) assemble formula
    OT_eps = (
        delta2
        + tr0
        + tr1
        - (epsilon / 2.0) * (tr_M - logdet_M + n * np.log(2.0) - 2.0 * n)
    )

    return OT_eps




