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


def sinkhorn_divergence_gaussian(m0, K0, m1, K1, epsilon):
    """
    Compute the entropic‐regularized 2‐Wasserstein cost OTₑ and the Sinkhorn divergence
    S₂ₑ between two Gaussians N(m0,K0) and N(m1,K1). Supports both 1D (scalar K0,K1)
    and multi‐dimensional (matrix K0,K1) cases.

    Returns
    -------
    sinkhorn_div : float
    ot_cost     : float
    """
    # detect 1D‐Gaussian case
    if np.isscalar(K0) and np.isscalar(K1):
        # scalar variances
        k0, k1 = float(K0)**2, float(K1)**2
        m0, m1 = float(m0), float(m1)
        n = 1
        def term_scalar(k_i, k_j):
            M = 1 + np.sqrt(1 + (epsilon**2 / 16) * (k_i * k_j))
            return (M
                    - np.log(M)
                    + n * np.log(2)
                    - 2 * n)
        def ot_pair(k_i, k_j, m_i, m_j):
            return ( (m_i - m_j)**2
                    + k_i + k_j
                   - epsilon * term_scalar(k_i, k_j) )
        ot_01 = ot_pair(k0, k1, m0, m1)
        ot_00 = ot_pair(k0, k0, m0, m0)
        ot_11 = ot_pair(k1, k1, m1, m1)
    else:
        # vector/matrix case
        m0 = np.atleast_1d(m0)
        m1 = np.atleast_1d(m1)
        K0 = np.atleast_2d(K0)
        K1 = np.atleast_2d(K1)
        n = m0.shape[0]
        I = np.eye(n)
        def Mij(Ki, Kj):
            return I + sqrtm(I + (epsilon**2 / 16) * (Ki @ Kj))
        def ot_pair(Ki, Kj, mi, mj):
            M = Mij(Ki, Kj)
            term = (np.trace(M)
                    - np.log(np.linalg.det(M))
                    + n * np.log(2)
                    - 2 * n)
            return (np.linalg.norm(mi - mj)**2
                    + np.trace(Ki) + np.trace(Kj)
                   - epsilon * term)
        ot_01 = ot_pair(K0, K1, m0, m1)
        ot_00 = ot_pair(K0, K0, m0, m0)
        ot_11 = ot_pair(K1, K1, m1, m1)

    sinkhorn_div = ot_01 - 0.5 * (ot_00 + ot_11)
    return sinkhorn_div, ot_01

