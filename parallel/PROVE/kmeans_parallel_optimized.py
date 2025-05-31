# kmeans_parallel_optimized.py
"""
31 May 2025
List of changes:
1. Limit the copy-paste of data: use global variables!
2. Use a worker initializer to pass the global variables to each worker
3. Return tuples instead of dataclasses
"""


import numpy as np
from multiprocessing import Pool
from dataclasses import dataclass # for clear outputs in the form of a class

@dataclass
class KMeansResult:
    assignments: np.ndarray
    final_centroids: np.ndarray
    loss: list

# Global variables
X_shared = None
def init_worker(X):
    global X_shared
    X_shared = X

def compute_chunk_assignments(index_range, centroids):
    global X_shared
    """
    Re - assigns the points to the cluster with closest centroid, then computes the new loss
    """
    start_idx, end_idx = index_range
    X_chunk = X_shared[start_idx:end_idx, :]
    distances = np.linalg.norm(X_chunk[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    assignments = np.argmin(distances, axis=1)
    loss = np.sum(distances[np.arange(len(assignments)), assignments])
    return assignments, loss

def parallel_k_means_optimized(X:np.ndarray, k:int, seed:int, n_iter_max:int, verbose = False):

    np.random.seed(seed)
    num_points, _ = X.shape
    assignments = np.zeros(shape = (num_points, ), dtype = int)
    WCSD_list = []

    # Parallel related variables
    N_PROCESSES = 8
    chunk_sizes = np.array_split(np.arange(num_points), N_PROCESSES)
    index_chunk_ranges = [(chunk[0], chunk[-1]+1) for chunk in chunk_sizes]
    """
    Centroid initialization
    """
    idxs = np.random.choice(np.arange(0, num_points), size = k, replace = False)
    centroids = X[idxs, :].copy() ## OPTIMIZATION 1.

    with Pool(processes = N_PROCESSES ,initializer = init_worker, initargs = (X,)) as pool:
        for n in range(n_iter_max):
            old_assignments = assignments.copy()
            """
            Compute current distances and update point assignments accordingly
            """
            
            chunk_results_list = pool.starmap(compute_chunk_assignments,\
                        [(range, centroids) \
                        for range in index_chunk_ranges ])
            assignments = np.concatenate([cr[0] for cr in chunk_results_list])
            wcsd = sum(cr[1] for cr in chunk_results_list)
            #assignments = np.concatenate([chunk_results_list[i].assignments for i in range(N_PROCESSES)])
            #wcsd = np.sum([chunk_results_list[i].loss for i in range(N_PROCESSES)])
            WCSD_list.append(wcsd)
            # Update assignments
            if np.all(old_assignments == assignments):
                if verbose:
                    print(f"Algorithm converged at iteration {n}/{n_iter_max}")
                break
            """
            Update centroids
            """
            for centroid_idx in range(k):
                # OPTIMIZATION 3 :one less "for loop"
                # OPTIMIZATION 4: use np.mean directly
                centroids[centroid_idx, :] = np.mean(X[(assignments == centroid_idx), :], axis = 0)
    """
    Compute final loss
    """
    #final_distances = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    #wcsd = np.sum(final_distances[np.arange(len(assignments)), assignments])
    #WCSD_list.append(wcsd)
    results = KMeansResult(assignments, centroids, WCSD_list)
    if n == n_iter_max and verbose:
        print(f"Algorithm did not converge in {n_iter_max} iterations.")
    return results