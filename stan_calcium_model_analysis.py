#!/usr/bin/env python
import os.path
import argparse
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from stan_helpers import StanSampleAnalyzer, calcium_ode, moving_average

def main():
    # unpack arguments
    args = get_args()
    result_dir = args.result_dir
    cell_id = args.cell_id
    filter_type = args.filter_type
    moving_average_window = args.moving_average_window
    t0 = args.t0
    use_summary = args.use_summary
    num_chains = args.num_chains
    warmup = args.warmup
    tasks = args.tasks
    show_progress = args.show_progress

    # initialize Stan analyzer
    y_ref = np.loadtxt("canorm_tracjectories.csv", delimiter=",")
    y_ref_cell = y_ref[cell_id, :]
    if filter_type == "moving_average":
        y_ref_cell = moving_average(y_ref_cell, window=moving_average_window)
        y_ref_cell = np.squeeze(y_ref_cell)
    y_ref_cell = y_ref_cell[t0:]
    y0 = np.array([0, 0, 0.7, y_ref_cell[t0]])
    t_end = 1000
    ts = np.linspace(t0, t_end, t_end - t0 + 1)
    param_names = ["sigma", "KonATP", "L", "Katp", "KoffPLC", "Vplc", "Kip3",
                   "KoffIP3", "a", "dinh", "Ke", "Be", "d1", "d5", "epr",
                   "eta1", "eta2", "eta3", "c0", "k3"]
    if use_summary:
        analyzer = StanSampleAnalyzer(result_dir, calcium_ode, ts, y0, 3,
                                    use_summary=use_summary,
                                    param_names=param_names, y_ref=y_ref_cell,
                                    show_progress=show_progress)
    else:
        analyzer = StanSampleAnalyzer(result_dir, calcium_ode, ts, y0, 3,
                                    num_chains=num_chains, warmup=warmup,
                                    param_names=param_names, y_ref=y_ref_cell,
                                    show_progress=show_progress)

    # run tasks
    if "all" in tasks:
        tasks = ["simulate_chains", "plot_parameters", "get_r_squared"]
    if "simulate_chains" in tasks:
        analyzer.simulate_chains()
    if "plot_parameters" in tasks:
        analyzer.plot_parameters()
    if "get_r_squared" in tasks:
        analyzer.get_r_squared()

def get_args():
    """parse command line arguments"""
    arg_parser = argparse.ArgumentParser(description="Analyze Stan sample "
                                         + "files.")
    arg_parser.add_argument("--result_dir", dest="result_dir", metavar="DIR",
                            type=str, required=True)
    arg_parser.add_argument("--cell_id", dest="cell_id", metavar="N", type=int,
                            default=0)
    arg_parser.add_argument("--filter_type", dest="filter_type",
                            choices=["none", "moving_average"], default="none")
    arg_parser.add_argument("--moving_average_window",
                            dest="moving_average_window", type=int, default=20)
    arg_parser.add_argument("--t0", dest="t0", metavar="T0", type=int,
                            default=200)
    arg_parser.add_argument("--use_summary", dest="use_summary", default=False,
                            action="store_true")
    arg_parser.add_argument("--num_chains", dest="num_chains", type=int,
                            default=4)
    arg_parser.add_argument("--warmup", dest="warmup", type=int, default=1000)
    arg_parser.add_argument("--tasks", dest="tasks", nargs="+",
                            choices=["all", "simulate_chains",
                                     "plot_parameters", "get_r_squared"],
                            default=["all"])
    arg_parser.add_argument("--show_progress", dest="show_progress",
                            default=False, action="store_true")

    return arg_parser.parse_args()

if __name__ == "__main__":
    main()
