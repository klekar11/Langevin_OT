import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def animate_coupling(
    X_traj: np.ndarray,
    Y_traj: np.ndarray,
    errors: np.ndarray,
    t: np.ndarray,
    T_map,
    filepath: str = 'coupling_animation.mp4',
    fps: int = 15,
    interval: int = 60,
    figsize=(11, 4),
    cmap='rainbow',
):
    """
    Animate the coupling trajectories and error convergence.

    Parameters
    ----------
    X_traj : (N+1, n) array
        Trajectory of X samples over time.
    Y_traj : (N+1, n) array
        Trajectory of Y samples over time.
    errors : (N+1,) array
        Mean-squared-error at each time step.
    t : (N+1,) array
        Corresponding time points.
    T_map : callable
        Optimal transport map function T(x).
    filepath : str
        File path for saving the animation (mp4).
    fps : int
        Frames per second for saving.
    interval : int
        Delay between frames in milliseconds for display.
    figsize : tuple
        Figure size for the animation.
    cmap : str
        Matplotlib colormap for particles.
    """
    num_samples = X_traj.shape[1]

    # Prepare OT curve
    x_curve = np.linspace(np.min(X_traj), np.max(X_traj), 600)
    y_curve = T_map(x_curve)

    # Figure and axes
    fig, (ax_xy, ax_err) = plt.subplots(1, 2, figsize=figsize)
    particle_colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, num_samples))

    # Left: scatter + OT curve
    ax_xy.plot(x_curve, y_curve, color='k', lw=1.5, label=r'optimal map $y=T(x)$')
    scatter = ax_xy.scatter([], [], s=8, alpha=0.65)
    time_text = ax_xy.text(0.02, 0.95, '', transform=ax_xy.transAxes,
                           va='top', ha='left', fontsize=9)
    ax_xy.set_xlabel(r'$X_t$')
    ax_xy.set_ylabel(r'$Y_t$')
    ax_xy.set_xlim(x_curve[0], x_curve[-1])
    ax_xy.set_ylim(y_curve.min(), y_curve.max())
    ax_xy.legend()

    # Right: error convergence
    ax_err.semilogy(t, errors, lw=1.4, color='C1')
    marker = ax_err.scatter([], [], color='red', zorder=3)
    ax_err.set_xlabel('time $t$')
    ax_err.set_ylabel(r'$\mathbb{E}[\,|X_t-Y_t|^2\,]$')
    ax_err.grid(True, which='both', ls=':')

    # Initialization for animation
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        marker.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return scatter, marker, time_text

    # Update function for animation
    def update(frame):
        pts = np.column_stack((X_traj[frame], Y_traj[frame]))
        scatter.set_offsets(pts)
        scatter.set_facecolor(particle_colors)

        marker.set_offsets([[t[frame], errors[frame]]])
        time_text.set_text(fr'$t = {t[frame]:.2f}$')
        return scatter, marker, time_text

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(t),
        init_func=init, interval=interval, blit=True
    )

    # Save and display
    ani.save(filepath, writer='ffmpeg', dpi=150, fps=fps)
    print(f'Animation saved as {filepath}')
    plt.tight_layout()
    plt.show()

    return ani


def save_scatter_snapshot(
    time_point: float,
    t: np.ndarray,
    X_traj: np.ndarray,
    Y_traj: np.ndarray,
    T_map,
    filename: str,
    figsize=(6, 4),
    cmap='rainbow'
):
    """
    Save and display a snapshot of the coupling scatter against the OT curve at a given time,
    with axes limited to x in [-3,3] and y in [-2,2].
    """
    # find nearest frame
    idx = np.argmin(np.abs(t - time_point))
    
    # OT curve
    x_curve = np.linspace(-3, 3, 600)
    y_curve = T_map(x_curve)
    # mask non-finite
    mask = np.isfinite(y_curve)
    x_curve = x_curve[mask]
    y_curve = y_curve[mask]
    
    # plot
    fig, ax = plt.subplots(figsize=figsize)
    n = X_traj.shape[1]
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, n))
    
    ax.plot(x_curve, y_curve, 'k-', lw=1.5)
    ax.scatter(X_traj[idx], Y_traj[idx], s=8, alpha=0.65, color=colors)
    ax.set_title(f'$t={t[idx]:.2f}$')
    ax.set_xlabel('$X_t$')
    ax.set_ylabel('$Y_t$')
    
    # force fixed axis limits
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 2)
    
    ax.legend()
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close(fig)
