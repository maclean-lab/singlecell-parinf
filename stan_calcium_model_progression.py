#!/usr/bin/env python
# Analyze how the calcium model change with cells
import os.path
import itertools
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# parameters in the calcium model
param_names = ["sigma", "KonATP", "L", "Katp", "KoffPLC", "Vplc", "Kip3",
               "KoffIP3", "a", "dinh", "Ke", "Be", "d1", "d5", "epr", "eta1",
               "eta2", "eta3", "c0", "k3"]
num_params = len(param_names)

# cells to analyze
cell_ids = [0, 3369, 1695, 61, 2623, 619, 1271, 4927, 613, 4305]
runs = [4, 1, 1, 1, 1, 1, 1, 1, 1, 1]
t0s = [221, 220, 230, 187, 214, 200, 200, 200, 200, 210]
chains = [[2], "all", "all", "all", "all", "all", "all", "all", "all", "all"]
num_cells = len(cell_ids)

# path of result
result_root = "../../result"
cell_dirs = [os.path.join(result_root,
                          "stan-calcium-model-hpc-cell-{}-{}".format(c, r))
             for c, r, in zip(cell_ids, runs)]
progression_dir = os.path.join(result_root, "stan-calcium-model-progression")

def main():
    plot_average_r_squared()
    plot_average_running_time()

def plot_average_r_squared():
    """plot average of R^2 for all pairs of parameters"""
    print("Plotting average R^2")

    # compute average of R^2
    average_r_squared = np.zeros((num_params, num_params, num_cells))
    for k in range(num_cells):
        cell_chains = [0, 1, 2, 3] if chains[k] == "all" else chains[k]

        for chain in cell_chains:
            r_squared_file = os.path.join(
                cell_dirs[k], "chain_{}_r_squared.csv".format(chain)
            )
            r_squared = pd.read_csv(r_squared_file, index_col=0)
            average_r_squared[:, :, k] = average_r_squared[:, :, k] + r_squared.to_numpy()

        average_r_squared[:, :, k] /= len(cell_chains)

    # make plots
    num_pairs = num_params * (num_params - 1) // 2
    num_rows, num_cols = 4, 2
    num_subplots_per_page = num_rows * num_cols
    num_pages = math.ceil(num_pairs / num_subplots_per_page)
    i, j = 0, 1
    figure_path = os.path.join(progression_dir, "r_squared_progression.pdf")
    with PdfPages(figure_path) as pdf:
        for page in range(num_pages):
            # set page size as US letter
            plt.figure(figsize=(8.5, 11))

            # set number of subplots in current page
            if page == num_pages - 1:
                num_subplots = (num_pairs - 1) % num_subplots_per_page + 1
            else:
                num_subplots = num_subplots_per_page

            # plot each pair of parameters
            for idx in range(num_subplots):
                plt.subplot(
                    num_rows, num_cols, idx + 1,
                    title="{} vs {}".format(param_names[i], param_names[j])
                )
                plt.plot(average_r_squared[i, j, :], "x")
                plt.xticks(np.arange(num_cells), cell_ids)
                plt.ylim((0, 1))

                # advance to next pair of parameters
                j += 1
                if j == num_params:
                    i += 1
                    j = i + 1

            plt.tight_layout()
            pdf.savefig()
            plt.close()

    """
    for i, j in itertools.combinations(range(num_params), 2):
        plt.clf()
        plt.plot(average_r_squared[i, j, :], "x")
        plt.xticks(np.arange(num_cells), cell_ids)
        plt.ylim((0, 1))
        plt.title("{} vs {}".format(param_names[i], param_names[j]))
        figure_name = os.path.join(
            progression_dir,
            "r_squared_{}_{}.png".format(param_names[i], param_names[j])
        )
        plt.savefig(figure_name)
        plt.close()
    """

def plot_average_running_time():
    average_time = np.zeros(num_cells)

    for k in range(num_cells):
        cell_chains = [0, 1, 2, 3] if chains[k] == "all" else chains[k]

        for chain in cell_chains:
            # load sample file
            sample_file = os.path.join(cell_dirs[k],
                                       "chain_{}.csv".format(chain))
            with open(sample_file, "r") as sf:
                lines = sf.readlines()
                words = lines[-2].split()
                average_time[k] += float(words[1]) / 60

        average_time[k] /= len(cell_chains)

    plt.clf()
    plt.plot(average_time, "x")
    plt.xticks(np.arange(num_cells), cell_ids)
    plt.xlabel("Cell IDs")
    plt.ylabel("Running time (min)")
    plt.savefig(os.path.join(progression_dir, "average_running_time.png"))
    plt.close()

if __name__ == "__main__":
    main()
