import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from dataclasses import dataclass # for clear outputs in the form of a class

@dataclass
class KMeansResult:
    assignments: np.ndarray
    final_centroids: np.ndarray
    loss: list




def compute_chunk_assignments(X_chunk, centroids):
    distances = np.linalg.norm(X_chunk[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    assignments = np.argmin(distances, axis=1)
    return assignments


def parallel_k_means(k: int, X: np.ndarray, n_iter_max: int, seed: int, verbose = False):
    """
    The only difference is in the computation of distances
    k: number of clusters
    X: data matrix with nrows = number of points, ncols = number of features
    n_max: maximum number of iterations

    The algorithm stops when either the assignations of the points do not change anymore,
    or n_max is reached.
    """
    np.random.seed(seed)
    num_points, _ = X.shape
    assignments = np.zeros(shape = (num_points, ), dtype = int)
    WCSD_list = []
    """
    Centroid initialization
    """
    idxs = np.random.choice(np.arange(0, num_points), size = k, replace = False)
    centroids = X[idxs, :].copy() ## OPTIMIZATION 1.

    for n in range(n_iter_max):
        old_assignments = assignments.copy()
        """
        Compute current distances and update point assignments accordingly
        """
        # distances is (N, k)
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2) # OPTIMIZATION 2.
        wcsd = np.sum(distances[np.arange(len(assignments)), assignments])
        WCSD_list.append(wcsd)
        # Update assignments
        assignments = np.argmin(distances, axis=1)
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
    final_distances = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    wcsd = np.sum(final_distances[np.arange(len(assignments)), assignments])
    WCSD_list.append(wcsd)
    results = KMeansResult(assignments, centroids, WCSD_list)
    if n == n_iter_max and verbose:
        print(f"Algorithm did not converge in {n_iter_max} iterations.")
    return results

results = serial_k_means_optimized(k = 3, X = data_numpy.data, n_iter_max = 10, seed = 1234)