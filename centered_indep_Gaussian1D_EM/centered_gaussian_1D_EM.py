import numpy as np

def centered_gaussian_1D_EM(X_0, Y_0, epsilon, N, T, num_samples, sigma1, sigma2):
    """
    Simulates a coupled SDE system and returns the trajectories and errors.

    Parameters:
    - X_0: float, initial condition for X
    - Y_0: float, initial condition for Y
    - epsilon: float, regularization parameter
    - N: int, number of time steps
    - T: float, final time
    - num_samples: int, number of sample paths (default 1000)
    - sigma1: float, standard deviation for X noise
    - sigma2: float, standard deviation for Y noise

    Returns:
    - errors: np.ndarray of shape (N+1,), mean squared error over time
    - X_traj: np.ndarray of shape (num_samples, N+1), trajectories of X
    - Y_traj: np.ndarray of shape (num_samples, N+1), trajectories of Y
    """

    Dt = T / N

    # Initialize trajectories
    X_traj = np.zeros((num_samples, N+1))
    Y_traj = np.zeros((num_samples, N+1))
    X_traj[:, 0] = X_0
    Y_traj[:, 0] = Y_0

    # Brownian increments
    W_gauss = np.sqrt(Dt * 2 * epsilon) * np.random.normal(0, 1, (num_samples, N))
    B_gauss = np.sqrt(Dt * 2 * epsilon) * np.random.normal(0, 1, (num_samples, N))

    # Correlation array
    correlations = np.zeros(N+1)
    correlations[0] = np.mean(X_traj[:, 0] * Y_traj[:, 0])

    for n in range(N):
        X = X_traj[:, n]
        Y = Y_traj[:, n]

        X_next = X + (Y - (correlations[n] + epsilon) * (1 / sigma1**2) * X) * Dt + W_gauss[:, n]
        Y_next = Y + (X - (correlations[n] + epsilon) * (1 / sigma2**2) * Y) * Dt + B_gauss[:, n]

        X_traj[:, n + 1] = X_next
        Y_traj[:, n + 1] = Y_next

        correlations[n + 1] = np.mean(X_next * Y_next)

    # Compute errors after full simulation
    errors = np.mean((X_traj - Y_traj)**2, axis=0)

    return errors, X_traj, Y_traj, correlations
