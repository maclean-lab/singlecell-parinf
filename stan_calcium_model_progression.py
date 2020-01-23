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
num_cells = len(cell_ids)
num_chains = 4
cell_meta = pd.DataFrame(index=cell_ids)
# cell_meta["run"] = [4, 1, 1, 1, 1, 1, 1, 1, 1, 1]
cell_meta["run"] = [4, "2-moving-average", "3-moving-average",
                    "3-moving-average", "2-moving-average", "2-moving-average",
                    "2-moving-average", "4-moving-average", "2-moving-average",
                    "2-moving-average"]
cell_meta["t0s"] = [221, 220, 230, 187, 214, 200, 200, 200, 200, 210]
# cell_meta["converged_chains"] = [[2], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3],
#                                  [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3],
#                                  [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
cell_meta["converged_chains"] = [[2], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3],
                                 [0, 1, 2, 3], [0, 1, 2, 3], [0, 2, 3],
                                 [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]

# path of result
result_root = "../../result"
for cell in cell_ids:
    cell_meta.loc[cell, "dir"] = os.path.join(
        result_root,
        "stan-calcium-model-hpc-cell-" \
            + "{}-{}".format(cell, cell_meta.loc[cell, "run"])
    )
progression_dir = os.path.join(result_root, "stan-calcium-model-progression")

def main():
    plot_r_squared()
    plot_average_running_time()
    plot_parameter_violin()

def plot_r_squared():
    """plot R^2 for all pairs of parameters"""
    print("Plotting R^2 between parameters...")

    # compute average of R^2
    r_squared_all = np.zeros((num_params, num_params, num_cells,
                              num_chains))
    for k, cell in enumerate(cell_ids):
        for chain in range(num_chains):
            r_squared_file = os.path.join(
                cell_meta.loc[cell, "dir"],
                "chain_{}_r_squared.csv".format(chain)
            )
            r_squared = pd.read_csv(r_squared_file, index_col=0)
            r_squared_all[:, :, k, chain] = r_squared.to_numpy()

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

                for chain in range(num_chains):
                    plt.plot(r_squared_all[i, j, :, chain], "x")

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
    print("Plotting average running time...")

    average_time = np.zeros(num_cells)
    for k in range(num_cells):
        for chain in range(num_chains):
            # load sample file
            sample_file = os.path.join(cell_meta.loc[cell_ids[k], "dir"],
                                       "chain_{}.csv".format(chain))
            with open(sample_file, "r") as sf:
                lines = sf.readlines()
                words = lines[-2].split()
                average_time[k] += float(words[1]) / 60

        average_time[k] /= num_chains

    plt.clf()
    plt.plot(average_time, "x")
    plt.xticks(np.arange(num_cells), cell_ids)
    plt.xlabel("Cell IDs")
    plt.ylabel("Running time (min)")
    plt.savefig(os.path.join(progression_dir, "average_running_time.png"))
    plt.close()

def plot_parameter_violin():
    """make violin plot of parameter distribution vs cell chain"""
    print("Making violin plots for parameters...")

    # initialize
    theta_0_col = 8  # column index of theta[0] in sample file
    # loaded sampled parameters are stored as  {param_name: list of samples}
    # where each sample in the list is sampled parameters of one cell
    all_samples = {}
    for param in param_names:
        all_samples[param] = []

    # load sample file of each cell
    for cell in cell_ids:
        cell_samples = []  # sampled parameters of each chain

        # load sample file of each chain
        for chain in cell_meta.loc[cell, "converged_chains"]:
            sample_file = os.path.join(
                cell_meta.loc[cell, "dir"], "chain_{}.csv".format(chain)
            )

            # get number of warm-up iterations from sample file
            with open(sample_file, "r") as sf:
                for line in sf:
                    if "warmup=" in line:
                        warmup = int(line.strip().split("=")[-1])
                        break

            # read sample file
            chain_samples = pd.read_csv(sample_file, index_col=False,
                                        comment="#")
            # store sampled parameters of this chain
            cell_samples.append(chain_samples.iloc[warmup:, theta_0_col - 1:])

        # combine and store sampled parameters from all chains
        cell_sample_combined = pd.concat(cell_samples)
        cell_sample_combined.columns = param_names
        for param in param_names:
            all_samples[param].append(cell_sample_combined[param].to_numpy())

    # make violin plots for each parameter
    num_rows, num_cols = 4, 2
    num_subplots_per_page = num_rows * num_cols
    num_pages = math.ceil(num_params / num_subplots_per_page)
    figure_path = os.path.join(progression_dir, "param_violin.pdf")
    with PdfPages(figure_path) as pdf:
        # generate each page
        for page in range(num_pages):
            # set page size as US letter
            plt.figure(figsize=(8.5, 11))

            # set number of subplots in current page
            if page == num_pages - 1:
                num_subplots = (num_cells - 1) % num_subplots_per_page + 1
            else:
                num_subplots = num_subplots_per_page

            # plot each parameter
            for idx in range(num_subplots):
                param_idx = page * num_subplots_per_page + idx
                plt.subplot(num_rows, num_cols, idx + 1,
                            title=param_names[param_idx])
                plt.violinplot(all_samples[param_names[param_idx]])
                plt.xticks(np.arange(1, num_cells + 1), cell_ids)

            plt.tight_layout()
            pdf.savefig()
            plt.close()

if __name__ == "__main__":
    main()
