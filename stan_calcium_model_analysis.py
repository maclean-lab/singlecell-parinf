#!/usr/bin/env python
import os.path
import argparse
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from stan_helpers import StanSessionAnalyzer, calcium_ode, moving_average

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
    integrator = args.integrator
    integrator_method = args.integrator_method

    # initialize Stan analyzer
    y = np.loadtxt("canorm_tracjectories.csv", delimiter=",")
    y = y[cell_id, :]
    t_end = y.size - 1
    # apply preprocessing to the trajectory if specified
    if filter_type == "moving_average":
        y = moving_average(y, window=moving_average_window)
        y = np.squeeze(y)
    ts = np.linspace(t0 + 1, t_end, t_end - t0)
    # downsample trajectories
    t_downsample = 300
    y = np.concatenate((y[0:t_downsample], y[t_downsample::10]))
    ts = np.concatenate((ts[0:t_downsample-t0], ts[t_downsample-t0::10]))
    y0 = np.array([0, 0, 0.7, y[t0]])
    y_ref = y[t0 + 1:]
    param_names = ["sigma", "KonATP", "L", "Katp", "KoffPLC", "Vplc", "Kip3",
                   "KoffIP3", "a", "dinh", "Ke", "Be", "d1", "d5", "epr",
                   "eta1", "eta2", "eta3", "c0", "k3"]
    if use_summary:
        analyzer = StanSessionAnalyzer(result_dir, use_summary=use_summary,
                                      param_names=param_names)
    else:
        analyzer = StanSessionAnalyzer(result_dir, num_chains=num_chains,
                                       warmup=warmup, param_names=param_names)

    # run tasks
    if "all" in tasks:
        tasks = ["simulate_chains", "plot_parameters", "get_r_squared"]
    if "simulate_chains" in tasks:
        integrator_params = {}
        if integrator == "vode":
            integrator_params["method"] = integrator_method

        analyzer.simulate_chains(
            calcium_ode, t0, ts, y0, 3, y_ref=y_ref,
            show_progress=show_progress, integrator=integrator,
            **integrator_params)
    if "plot_parameters" in tasks:
        analyzer.plot_parameters()
    if "get_r_squared" in tasks:
        analyzer.get_r_squared()

def get_args():
    """parse command line arguments"""
    arg_parser = argparse.ArgumentParser(
        description="Analyze Stan sample files.")
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
    arg_parser.add_argument("--integrator", dest="integrator", type=str,
                            default="dopri5")
    arg_parser.add_argument("--integrator_method", dest="integrator_method",
                            type=str, default="adams")

    return arg_parser.parse_args()

if __name__ == "__main__":
    main()
