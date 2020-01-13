#!/usr/bin/env python
import os.path
import numpy as np
import matplotlib.pyplot as plt
from stan_helpers import get_prior_from_sample_file

param_names = ["KonATP", "L", "Katp", "KoffPLC", "Vplc", "Kip3",
               "KoffIP3", "a", "dinh", "Ke", "Be", "d1", "d5", "epr",
               "eta1", "eta2", "eta3", "c0", "k3"]
num_params = len(param_names)
cell_ids = [0, 3369, 1695, 61, 2623, 619, 1271, 4927, 613, 4305]
num_cells = len(cell_ids)

def main():
    result_dir_root = "../../result"
    sample_files = [
        "stan-calcium-model-hpc-cell-0-4/chain_2.csv",
        "stan-calcium-model-hpc-cell-3369-1/chain_0.csv",
        "stan-calcium-model-hpc-cell-1695-1/chain_0.csv",
        "stan-calcium-model-hpc-cell-61-1/chain_0.csv",
        "stan-calcium-model-hpc-cell-2623-1/chain_0.csv",
        "stan-calcium-model-hpc-cell-619-1/chain_0.csv",
        "stan-calcium-model-hpc-cell-1271-1/chain_0.csv",
        "stan-calcium-model-hpc-cell-4927-1/chain_0.csv",
        "stan-calcium-model-hpc-cell-613-1/chain_0.csv",
        "stan-calcium-model-hpc-cell-4305-1/chain_0.csv",
    ]

    # get priors
    prior_means = np.empty((num_params, num_cells))
    prior_stds = np.empty((num_params, num_cells))
    for i in range(num_cells):
        prior_mean, prior_std = get_prior_from_sample_file(
            os.path.join(result_dir_root, sample_files[i])
        )
        prior_means[:, i] = prior_mean
        prior_stds[:, i] = prior_std

    plot_prior_changes(prior_means,
                       os.path.join(result_dir_root, "prior_mean_change.png"))
    plot_prior_changes(prior_stds,
                       os.path.join(result_dir_root, "prior_std_change.png"))

def plot_prior_changes(prior_stats, figure_name):
    """plot changes in prior mean or standard deviation"""
    # initialize the figure
    plt.clf()
    plt.figure(figsize=(6, num_params * 2))

    # plot change of each parameter
    for i in range(num_params):
        plt.subplot(num_params, 1, i + 1)
        plt.plot(prior_stats[i, :], "x")
        plt.xticks(np.arange(num_cells), cell_ids)
        plt.title(param_names[i])

    plt.tight_layout()

    # save the figure
    plt.savefig(figure_name)
    plt.close()

if __name__ == "__main__":
    main()
