import numpy as np
from typing import Tuple, Optional

def frobinnerproduct(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius inner product: sum of elementwise products."""
    return np.sum(A * B)

def round_transpoly(X: np.ndarray, r: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Round matrix X onto the transport polytope U(r,c) via
    two-step scaling and a final rank-one correction.
    """
    M = X.copy()
    n = M.shape[0]
    # 1) Row clipping
    row_sums = M.sum(axis=1)
    for i in range(n):
        α = min(1.0, r[i] / row_sums[i])
        M[i, :] *= α
    # 2) Column clipping
    col_sums = M.sum(axis=0)
    for j in range(n):
        β = min(1.0, c[j] / col_sums[j])
        M[:, j] *= β
    # 3) Rank-one correction
    row_sums = M.sum(axis=1)
    col_sums = M.sum(axis=0)
    err_r = row_sums - r
    err_c = col_sums - c
    M += np.outer(err_r, err_c) / np.sum(np.abs(err_r))
    return M

def sinkhorn(
    K: np.ndarray,
    r: np.ndarray,
    c: np.ndarray,
    T: int,
    compute_otvals: bool = False,
    C: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Classical Sinkhorn with kernel input K.

    Args:
      K              – kernel matrix, typically exp(−η C)
      r              – target row-marginal (n,)
      c              – target col-marginal (n,)
      T              – number of Sinkhorn iterations
      compute_otvals – if True, track OT cost of rounded iterates
      C              – cost matrix (required if compute_otvals=True)

    Returns:
      P      – final scaled coupling
      err    – ℓ₁-violation of marginals at each step (length T+1)
      otvals – cost 〈rounded P, C〉 over iterations (or None)
    """
    P = K.copy()
    err = np.zeros(T + 1)

    # initial marginal‐violation
    rP = P.sum(axis=1)
    cP = P.sum(axis=0)
    err[0] = np.linalg.norm(rP - r, 1) + np.linalg.norm(cP - c, 1)

    otvals: Optional[np.ndarray] = None
    if compute_otvals:
        if C is None:
            raise ValueError("Must provide cost matrix C when compute_otvals=True")
        otvals = np.zeros(T + 1)
        otvals[0] = frobinnerproduct(round_transpoly(P, r, c), C)

    for t in range(1, T + 1):
        if t % 2 == 1:
            # row-scaling
            rP = P.sum(axis=1)
            P *= (r / rP)[:, None]
        else:
            # column-scaling
            cP = P.sum(axis=0)
            P *= (c / cP)

        # track violation
        rP = P.sum(axis=1)
        cP = P.sum(axis=0)
        err[t] = np.linalg.norm(rP - r, 1) + np.linalg.norm(cP - c, 1)

        # optional cost evaluation
        if compute_otvals:
            otvals[t] = frobinnerproduct(round_transpoly(P, r, c), C)  # type: ignore

    return P, err, otvals
