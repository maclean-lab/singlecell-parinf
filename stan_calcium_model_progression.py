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

    # get cell list
    cell_list = pd.read_csv(cell_list_path, delimiter="\t", index_col=False)
    cell_list = cell_list["Cell"].to_numpy()
    first_cell_order = np.where(cell_list == first_cell)[0][0]
    last_cell_order = np.where(cell_list == last_cell)[0][0]
    cell_list = cell_list[first_cell_order:last_cell_order]
    cell_dirs = ["cell-{:04d}".format(cell_id) for cell_id in cell_list]
    param_names = ["sigma", "KonATP", "L", "Katp", "KoffPLC", "Vplc", "Kip3",
                   "KoffIP3", "a", "dinh", "Ke", "Be", "d1", "d5", "epr",
                   "eta1", "eta2", "eta3", "c0", "k3"]
    analyzer = StanMultiSessionAnalyzer(cell_list, result_root, cell_dirs,
                                        use_summary=True,
                                        param_names=param_names)
    analyzer.plot_parameter_violin()

def get_args():
    """Parse command line arguments"""
    arg_parser = argparse.ArgumentParser(
        description="Analyze sampling result from a list of cells"
    )
    arg_parser.add_argument("--cell_list", dest="cell_list", type=str,
                            required=True)
    arg_parser.add_argument("--first_cell", dest="first_cell", type=int,
                            required=True)
    arg_parser.add_argument("--last_cell", dest="last_cell", type=int,
                            required=True)
    arg_parser.add_argument("--result_root", dest="result_root", type=str,
                            required=True)

    return arg_parser.parse_args()

if __name__ == "__main__":
    main()
