#!/usr/bin/env python
import os.path
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stan_helpers import moving_average

def main():
    # get filter option from command line
    arg_parser = argparse.ArgumentParser(
        description="Get distances between trajectories of consecutive cells.")
    arg_parser.add_argument("--filter_type", dest="filter_type", type=str,
                            default=None)
    arg_parser.add_argument("--cell_list", dest="cell_list", type=str,
                            default=None)
    arg_parser.add_argument("--t0", dest="t0", type=int, default=200)
    arg_parser.add_argument("--output_dir", dest="output_dir", type=str,
                            required=True)
    args = arg_parser.parse_args()

    # load trajectories
    print("Loading trajectories")
    y_raw = np.loadtxt("canorm_tracjectories.csv", delimiter=",")

    # apply filter
    if args.filter_type == "moving_average":
        y = moving_average(y_raw)
    else:
        print("No filter specified or unknown type of filter. Using raw "
              + "trajectories")

        y = y_raw

    # reorder cells if given a list of cells
    if args.cell_list:
        cell_list = pd.read_csv(args.cell_list, delimiter="\t", index_col=False)
        y = y[cell_list["Cell"], ]
        cell_names = [f"cell {cell_id}" for cell_id in cell_list["Cell"]]
    else:
        cell_names = [f"cell {cell_id}" for cell_id in range(y.shape[0])]

    # compute distances
    cell_dists = np.linalg.norm(y[:-1, args.t0:] - y[1:, args.t0:], ord=2,
                                axis=1)

    # save results
    cell_dist_table = pd.DataFrame()
    cell_dist_table["From"] = cell_names[:-1]
    cell_dist_table["To"] = cell_names[1:]
    cell_dist_table["Distance"] = cell_dists
    cell_dist_table.to_csv(
        os.path.join(args.output_dir, "trajectorty_dist_table.csv"))

    plt.clf()
    plt.hist(cell_dists, bins=20)
    plt.xlabel("Trajectory distance between consecutive cells")
    plt.ylabel("# pairs of consecutive cells")
    plt.savefig(os.path.join(args.output_dir, "trajectory_dist_hist.png"))
    plt.close()

if __name__ == "__main__":
    main()
