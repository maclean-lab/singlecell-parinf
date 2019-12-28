#!/usr/bin/env python
import os.path
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from stan_helpers import moving_average

def plot_trajectories(y: np.ndarray, figure_path: str):
    """plot all trajectories in one PDF file"""
    num_cells = y.shape[0]
    # specify page layout
    num_rows, num_cols = 4, 2
    num_subplots_per_page = num_rows * num_cols
    num_pages = math.ceil(num_cells / num_subplots_per_page)

    print("Generating plots for all trajectories...")
    with PdfPages(figure_path) as pdf:
        # generate each page
        for page in tqdm(range(num_pages)):
            # set page size as US letter
            plt.figure(figsize=(8.5, 11))

            # set number of subplots in current page
            if page == num_pages - 1:
                num_subplots = (num_cells - 1) % num_subplots_per_page + 1
            else:
                num_subplots = num_subplots_per_page

            # plot each trajectory
            for idx in range(num_subplots):
                cell_id = page * num_subplots_per_page + idx
                plt.subplot(num_rows, num_cols, idx + 1,
                            title="cell {}".format(cell_id))
                plt.plot(y[cell_id, :])

            plt.tight_layout()
            pdf.savefig()
            plt.close()

def main():
    # get filter option from command line
    arg_parser = argparse.ArgumentParser(description="Plot all trajectories.")
    arg_parser.add_argument("--filter_type", dest="filter_type", type=str,
                            default=None)
    args = arg_parser.parse_args()

    # load trajectories
    print("Loading trajectories")
    y_raw = np.loadtxt("canorm_tracjectories.csv", delimiter=",")

    # apply filter
    if args.filter_type == "moving_average":
        print("Running moving average filter...")

        y = moving_average(y_raw)
        figure_path = "../../result/trajectories_moving_average.pdf"
    else:
        print("No filter specified or unknown type of filter. Using raw " +
              "trajectories.")

        y = y_raw
        figure_path = "../../result/trajectories_raw.pdf"

    # make the trajectory plot
    plot_trajectories(y, figure_path)
    print("Trajectory plot saved.")

if __name__ == "__main__":
    main()
