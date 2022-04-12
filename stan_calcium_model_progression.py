#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd
from stan_helpers import StanMultiSessionAnalyzer

def main():
    args = get_args()
    cell_list_path = args.cell_list
    first_cell = args.first_cell
    last_cell = args.last_cell
    result_root = args.result_root
    param_mask = args.param_mask
    tasks = args.tasks

    # get cell list
    cell_list = pd.read_csv(cell_list_path, delimiter="\t", index_col=False)
    cell_list = cell_list["Cell"].to_numpy()
    first_cell_order = np.where(cell_list == first_cell)[0][0]
    last_cell_order = np.where(cell_list == last_cell)[0][0]
    cell_list = cell_list[first_cell_order:last_cell_order + 1]
    cell_dirs = [f"cell-{cell_id:04d}" for cell_id in cell_list]
    param_names = ["sigma", "KonATP", "L", "Katp", "KoffPLC", "Vplc", "Kip3",
                   "KoffIP3", "a", "dinh", "Ke", "Be", "d1", "d5", "epr",
                   "eta1", "eta2", "eta3", "c0", "k3"]
    # filter parameters
    if param_mask:
        param_names = [param_names[i + 1] for i, mask in enumerate(param_mask)
                       if mask == "1"]
        param_names = ["sigma"] + param_names

    analyzer = StanMultiSessionAnalyzer(cell_list, result_root, cell_dirs,
                                        sample_source="arviz_inf_data",
                                        param_names=param_names)
    if "all" in tasks:
        tasks = ["plot_parameter_violin", "plot_rhat"]
    if "plot_parameter_violin" in tasks:
        print("Making violin plots for parameters...")
        analyzer.plot_parameter_violin()
    if "plot_rhat" in tasks:
        print("Making R^hat plots for parameters and log postereior...")
        analyzer.plot_rhats()

def get_args():
    """Parse command line arguments"""
    arg_parser = argparse.ArgumentParser(
        description="Analyze sampling result from a list of cells")
    arg_parser.add_argument("--cell_list", type=str, required=True)
    arg_parser.add_argument("--first_cell", type=int, required=True)
    arg_parser.add_argument("--last_cell", type=int, required=True)
    arg_parser.add_argument("--result_root", type=str, required=True)
    arg_parser.add_argument("--param_mask", type=str, default=None)
    arg_parser.add_argument("--tasks", nargs="+", default="all",
                            choices=["all", "plot_parameter_violin",
                                     "plot_rhat"])

    return arg_parser.parse_args()

if __name__ == "__main__":
    main()
