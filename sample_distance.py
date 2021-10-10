import itertools
from typing import List

import numpy as np
import scipy.stats
import scipy.spatial
from sklearn.neighbors import NearestNeighbors

def get_kl_nn(posterior_samples: List[np.ndarray], method: str = 'yao',
              k: int = 2, nan_sub: float = np.log(2), random_seed: int = 0,
              verbose: bool = False) -> np.ndarray:
    """Compute KL divergences between pairs of posterior samples.

    For each pair of posterior samples, KL(P || Q) is computed, where P is the
    probability that nearest data points of a data point come from the other
    sample for each, and Q is the probability under null condition (i.e.
    pair of identical posterior samples). Original implementation in MATLAB:
    https://github.com/wollmanlab/ODEparamFitting_ABCSMC/blob/master/estimateKLdivergenceBasedOnNN.m

    Args:
        posterior_samples (List[np.ndarray]): List of posterior samples.
            Each sample is a matrix of size (sample_size, num_params).
        method (str, optional): Method for getting density estimate from
            nearest neighbors. Supported methods are 'yao', 'neighbor_fraction',
            and 'neighbor_any'. Defaults to 'yao'.
        k (int, optional): Number of nearest neighbors to find.  Defaults to 2.
        nan_sub (float, optional): Value used to substitute NaN. Default to
            log(2).
        random_seed (int, optional): Seed for random number generator. Defaults
            to 0.
        verbose (bool, optional): Print information if True. Defaults to False.

    Returns:
        np.ndarray: KL divergence between pairs of posterior samples
    """
    bit_generator = np.random.MT19937(random_seed)
    rng = np.random.default_rng(bit_generator)

    num_samples = len(posterior_samples)
    D = np.ones((num_samples, num_samples))
    if method == 'yao':
        nn = NearestNeighbors(n_neighbors=2, algorithm='kd_tree')
    else:
        nn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree')

    # determine distribution of nearest neighbors between pairs of samples
    for i in range(num_samples):
        D[i, i] = 0.5

        for j in range(i + 1, num_samples):
            if verbose:
                print(f'\rFinding nearest neighbors for samples {i} and {j}',
                      end='', flush=False)

            n_i = posterior_samples[i].shape[0]
            n_j = posterior_samples[j].shape[0]
            n = min(n_i, n_j)
            P_i = posterior_samples[i][rng.choice(n_i, size=n), :]
            P_j = posterior_samples[j][rng.choice(n_j, size=n), :]
            P_ij = np.vstack((P_i, P_j))

            nn.fit(P_ij)
            _, nn_indices = nn.kneighbors(P_ij)

            if method == 'yao':
                nn_indices = nn_indices[:, 1]
                D[i, j] = D[j, i] = np.mean(np.concatenate(
                    ((nn_indices[:n] < n).astype(int),
                     (nn_indices[n:] >= n).astype(int))
                ))
            elif method == 'neighbor_fraction':
                D[i, j] = D[j, i] = np.mean(np.concatenate(
                    (np.mean(nn_indices[:n, :] < n, axis=1),
                     np.mean(nn_indices[n:, :] >= n, axis=1))
                ))
            else:
                D[i, j] = D[j, i] = np.mean(np.concatenate(
                    (np.all(nn_indices[:n, :] < n, axis=1).astype(int),
                     np.all(nn_indices[n:, :] >= n, axis=1).astype(int))
                ))

    if verbose:
        print()

    # compute KL from D
    def kl_scalar(x):
        if x == 0 or x == 1:
            return nan_sub

        # D * 2 = D / (1/2), where 1/2 is the probablity for null condition
        return x * np.log(x * 2) + (1 - x) * np.log((1 - x) * 2)

    kl_vector = np.vectorize(kl_scalar)
    KL = kl_vector(D)

    if verbose:
        print('KL distances computed for all pairs of samples')

    return KL

def get_jensen_shannon(posterior_samples: List[np.ndarray],
                       subsample_size: int = 1000, random_seed: int = 0,
                       verbose: bool = False) -> float:
    """Compute Jensen-Shannon distances between pairs of samples.

    Args:
        posterior_samples (List[np.ndarray]): List of posterior samples.
            Each sample is a matrix of size (sample_size, num_params).
        subsample_size (int, optional): Size of subsample to be used for
            density estimation. Defaults to 1000.
        random_seed (int, optional): Seed for random number generator.
            Defaults to 0.
        verbose (bool, optional): Print information if True. Defaults to False.

    Returns:
        float: Jensen-Shannon distances between pairs of posterior samples
    """
    bit_generator = np.random.MT19937(random_seed)
    rng = np.random.default_rng(bit_generator)

    num_samples = len(posterior_samples)
    num_params = posterior_samples[0].shape[1]
    js_dists = np.empty((num_samples, num_samples))

    for i, j in itertools.combinations_with_replacement(range(num_samples), 2):
        if verbose:
            print('Computing Jensen-Shannon distance between sample '
                  f'{i:04d} and sample {j:04d}...')

        sample_i = posterior_samples[i]
        sample_j = posterior_samples[j]
        sample_min = np.minimum(np.amin(sample_i, axis=0),
                                np.amin(sample_j, axis=0))
        sample_max = np.maximum(np.amax(sample_i, axis=0),
                                np.amax(sample_j, axis=0))
        estimation_points = \
            rng.random(size=(subsample_size, num_params)) \
                * (sample_max - sample_min) + sample_min

        kernel_i = scipy.stats.gaussian_kde(sample_i.T)
        density_i = kernel_i(estimation_points.T)
        kernel_j = scipy.stats.gaussian_kde(sample_j.T)
        density_j = kernel_j(estimation_points.T)

        js_ij =  scipy.spatial.distance.jensenshannon(density_i, density_j)
        if np.isnan(js_ij):
            js_dists[i, j] = js_dists[j, i] = 1.0
        else:
            js_dists[i, j] = js_dists[j, i] = js_ij

    if verbose:
        print('Jensen-Shannon distances computed for all pairs of samples')

    return js_dists
