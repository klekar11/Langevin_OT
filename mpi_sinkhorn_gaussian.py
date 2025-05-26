import numpy as np
import ot
import os
import matplotlib.pyplot as plt
import time
from mpi4py import MPI

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Start total timing
total_start = time.time()

# Define problem
mean1 = np.array([0, 0, 0, 0])
cov1 = np.diag([1, 1, 1, 1])
mean2 = np.array([10, -5, 3, 7])
cov2 = np.diag([2, 2, 2, 2])

sample_sizes = [10**i for i in range(1, 5)]  # [10, 100, 1000, 10000]
epsilons = [0.4, 0.7, 1.0, 5, 10, 100]

# Distribute sample sizes among ranks
local_sample_sizes = sample_sizes[rank::size]
local_results = {eps: [] for eps in epsilons}

# Start timing per process
local_start = time.time()

# Compute on assigned sample sizes
np.random.seed(42 + rank)  # different seed per rank
for n in local_sample_sizes:
    X = np.random.multivariate_normal(mean1, cov1, size=n)
    Y = np.random.multivariate_normal(mean2, cov2, size=n)
    BW_dist = ot.solve_sample(X, Y, method='gaussian').value

    for eps in epsilons:
        W = ot.bregman.empirical_sinkhorn2(X, Y, reg=eps)
        rel_error = abs(W - BW_dist) / abs(BW_dist)
        local_results[eps].append((n, rel_error))

local_end = time.time()
local_time = local_end - local_start

# Print each process's time
print(f"[Rank {rank}] Local computation time: {local_time:.2f} seconds")

# Gather results at root
gathered = comm.gather(local_results, root=0)

# End total timing at root
total_end = time.time()

if rank == 0:
    # Merge results from all processes
    final_results = {eps: [] for eps in epsilons}
    for proc_results in gathered:
        for eps in epsilons:
            final_results[eps].extend(proc_results[eps])
    
    # Sort results by sample size
    for eps in final_results:
        final_results[eps] = sorted(final_results[eps], key=lambda x: x[0])
    
    # Plot
    for eps in epsilons:
        sizes = [x[0] for x in final_results[eps]]
        errors = [x[1] for x in final_results[eps]]
        plt.loglog(sizes, errors, marker='x', linestyle='-', label=fr"$\epsilon$ = {eps}")

    plt.xlabel(r"$N$")
    plt.ylabel(r"$\frac{|\mathcal{W}_{\mathrm{Sink}} - \mathcal{W}_{\mathrm{GBW}}|}{\mathcal{W}_{\mathrm{GBW}}}$")
    plt.title("Relative Error vs Sample Size for Different ε (MPI)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print total time
    total_time = total_end - total_start
    print(f"[Rank 0] Total elapsed time: {total_time:.2f} seconds")
