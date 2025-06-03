import numpy as np

def epsilon_c(alpha_u, alpha_v, beta_u, beta_v):
    num = (1.0/alpha_u) * (1.0/alpha_v) - (1.0/beta_u) * (1.0/beta_v)
    den = np.sqrt((1.0/alpha_u + 1.0/beta_u) * (1.0/alpha_v + 1.0/beta_v))
    return num / den

def kappa_X_given_Y(alpha_u, beta_v, eps):
    return np.sqrt(4 * alpha_u / (eps**2 * beta_v) + alpha_u**2) + alpha_u

def kappa_Y_given_X(alpha_v, beta_u, eps):
    return np.sqrt(4 * alpha_v / (eps**2 * beta_u) + alpha_v**2) + alpha_v

def rate_r(kxgy, kygx, eps):
    return eps * (kxgy + kygx) * (1.0 - 4.0 / (eps**2 * kxgy * kygx))


def simulate_projected_langevin(eps, dt, X0, Y0, num_steps):
    N = X0.shape[0]
    sqrt2eps_dt = np.sqrt(2 * eps * dt)
    X_traj = np.empty((num_steps + 1, N))
    Y_traj = np.empty((num_steps + 1, N))
    mean_diff2 = np.empty(num_steps + 1)
    X = X0.copy()
    Y = Y0.copy()
    X_traj[0] = X
    Y_traj[0] = Y
    mean_diff2[0] = np.mean((X - Y)**2)
    rng = np.random.default_rng()
    for k in range(1, num_steps + 1):
        mX = X.mean()
        mY = Y.mean()
        dX_drift = (Y - mY) - eps * grad_U(X)
        dY_drift = (X - mX) - eps * grad_V(Y)
        noise_X = sqrt2eps_dt * rng.standard_normal(N)
        noise_Y = sqrt2eps_dt * rng.standard_normal(N)
        X = X + dX_drift * dt + noise_X
        Y = Y + dY_drift * dt + noise_Y
        X_traj[k] = X
        Y_traj[k] = Y
        mean_diff2[k] = np.mean((X - Y)**2)
    return X_traj, Y_traj, mean_diff2

def grad_U(x):
    return x + 0.1 * np.sin(2 * x)

def grad_V(y):
    return 3.0 * y + 0.1 * np.sin(2 * y)

