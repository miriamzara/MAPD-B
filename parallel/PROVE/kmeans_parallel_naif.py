# kmeans_parallel_naif.py

"""
31 May 2025
A first, basic version of kmeans
It is actually slower than the optimized serial version of kmeans
This was a first working attempt. Look at file kmeans_parallel_optimized.py
to see refinements.
"""


import numpy as np
from multiprocessing import Pool
from dataclasses import dataclass # for clear outputs in the form of a class

@dataclass
class KMeansResult:
    assignments: np.ndarray
    final_centroids: np.ndarray
    loss: list

@dataclass
class ChunkResults:
    assignments: np.array
    loss: float

def compute_chunk_assignments(X_chunk, centroids):
    """
    Re - assigns the points to the cluster with closest centroid, then computes the new loss
    """
    distances = np.linalg.norm(X_chunk[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    assignments = np.argmin(distances, axis=1)
    loss = np.sum(distances[np.arange(len(assignments)), assignments])
    chunk_results = ChunkResults(assignments, loss)
    return chunk_results

def parallel_k_means_naif(X:np.ndarray, k:int, seed:int, n_iter_max:int, verbose = False):

    np.random.seed(seed)
    num_points, _ = X.shape
    assignments = np.zeros(shape = (num_points, ), dtype = int)
    WCSD_list = []

    # Parallel related variables
    N_PROCESSES = 8
    X_chunks_list = np.array_split(X, N_PROCESSES)
    """
    Centroid initialization
    """
    idxs = np.random.choice(np.arange(0, num_points), size = k, replace = False)
    centroids = X[idxs, :].copy() ## OPTIMIZATION 1.

    with Pool(N_PROCESSES) as pool:
        for n in range(n_iter_max):
            old_assignments = assignments.copy()
            """
            Compute current distances and update point assignments accordingly
            """
            
            chunk_results_list = pool.starmap(compute_chunk_assignments,\
                        [(X_chunks_list[i], centroids) \
                        for i in range(N_PROCESSES) ])
            assignments = np.concatenate([chunk_results_list[i].assignments for i in range(N_PROCESSES)])
            wcsd = np.sum([chunk_results_list[i].loss for i in range(N_PROCESSES)])
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