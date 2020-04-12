#!/usr/bin/env python
import os.path
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stan_helpers import moving_average, pdf_multi_plot

def main():
    # get filter option from command line
    arg_parser = argparse.ArgumentParser(description="Plot all trajectories.")
    arg_parser.add_argument("--filter_type", dest="filter_type", type=str,
                            default=None)
    arg_parser.add_argument("--cell_list", dest="cell_list", type=str,
                            default=None)
    args = arg_parser.parse_args()

    # load trajectories
    print("Loading trajectories")
    y_raw = np.loadtxt("canorm_tracjectories.csv", delimiter=",")

    figure_name = "trajectories"

    # apply filter
    if args.filter_type == "moving_average":
        y = moving_average(y_raw)
        figure_name += "_moving_average"
    else:
        print("No filter specified or unknown type of filter. Using raw "
              + "trajectories")

        y = y_raw
        figure_name += "_raw"

    cell_names = [f"cell {cell_id}" for cell_id in range(y.shape[0])]

    # Reorder cells if given a list of cells
    if args.cell_list:
        cell_list = pd.read_csv(args.cell_list, delimiter="\t", index_col=False)
        y = y[cell_list["Cell"], ]
        figure_name += "_ordered"
        cell_names = [f"cell {cell_id}" for cell_id in cell_list["Cell"]]

    figure_path = os.path.join("../../result", figure_name + ".pdf")

    # make the trajectory plot
    pdf_multi_plot(plt.plot, y, figure_path, titles=cell_names,
                   show_progress=True)
    print("Trajectory plot saved")

if __name__ == "__main__":
    main()
