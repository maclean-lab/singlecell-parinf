#!/usr/bin/env python
import os.path
import argparse
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import calcium_models
from stan_helpers import StanSessionAnalyzer, load_trajectories

def main():
    # unpack arguments
    args = get_args()
    result_dir = args.result_dir
    cell_id = args.cell_id
    filter_type = args.filter_type
    moving_average_window = args.moving_average_window
    var_mask = args.var_mask
    param_mask = args.param_mask
    ode_variant = args.ode_variant
    t0 = args.t0
    downsample_offset = args.downsample_offset
    stan_operation = args.stan_operation
    sample_source = args.sample_source
    tasks = args.tasks
    show_progress = args.show_progress
    integrator = args.integrator
    integrator_method = args.integrator_method

    # initialize Stan analyzer
    y, y0_ca, ts = load_trajectories(
        t0, filter_type=filter_type,
        moving_average_window=moving_average_window,
        downsample_offset=downsample_offset)
    y = y[cell_id, :]
    y0_ca = y0_ca[cell_id]
    y0 = np.array([0, 0, 0.7, y0_ca])
    y_ref = [None, None, None, y]

    param_names = ["sigma", "KonATP", "L", "Katp", "KoffPLC", "Vplc", "Kip3",
                   "KoffIP3", "a", "dinh", "Ke", "Be", "d1", "d5", "epr",
                   "eta1", "eta2", "eta3", "c0", "k3"]
    # filter parameters
    if param_mask:
        param_names = [param_names[i + 1] for i, mask in enumerate(param_mask)
                       if mask == "1"]
        param_names = ["sigma"] + param_names

    var_names = ["PLC", "IP3", "h", "Ca"]
    # filter variables
    if var_mask:
        var_names = [var_names[i] for i, mask in enumerate(var_mask)
                     if mask == "1"]
        y0 = np.array(
            [y0[i] for i, mask in enumerate(var_mask) if mask == "1"])
        y_ref = [y_ref[i] for i, mask in enumerate(var_mask) if mask == "1"]

    # get ODE function
    calcium_ode = getattr(calcium_models, "calcium_ode_" + ode_variant)

    analyzer = StanSessionAnalyzer(result_dir, stan_operation=stan_operation,
                                   sample_source=sample_source,
                                   param_names=param_names)

    # run tasks
    if "all" in tasks:
        tasks = ["simulate_chains", "plot_parameters", "get_r_squared"]
    if "simulate_chains" in tasks:
        integrator_params = {}
        if integrator == "vode":
            integrator_params["method"] = integrator_method

        _ = analyzer.simulate_chains(
            calcium_ode, 0, ts, y0, y_ref=y_ref, show_progress=show_progress,
            var_names=var_names, integrator=integrator, **integrator_params)
    if "plot_parameters" in tasks:
        analyzer.plot_parameters()
    if "get_r_squared" in tasks:
        analyzer.get_r_squared()

def get_args():
    """parse command line arguments"""
    arg_parser = argparse.ArgumentParser(
        description="Analyze Stan sample files.")
    arg_parser.add_argument("--result_dir", type=str, required=True)
    arg_parser.add_argument("--cell_id", type=int, default=0)
    arg_parser.add_argument("--var_mask", type=str, default=None)
    arg_parser.add_argument("--param_mask", type=str, default=None)
    arg_parser.add_argument("--ode_variant", type=str,required=True)
    arg_parser.add_argument("--filter_type", default=None,
                            choices=[None, "moving_average"])
    arg_parser.add_argument("--moving_average_window", type=int, default=20)
    arg_parser.add_argument("--t0", type=int, default=200)
    arg_parser.add_argument("--downsample_offset", type=int, default=300)
    arg_parser.add_argument("--stan_operation", type=str, default="sampling")
    arg_parser.add_argument("--sample_source", type=str,
                            default="arviz_inf_data")
    arg_parser.add_argument("--num_chains", type=int, default=4)
    arg_parser.add_argument("--warmup", type=int, default=1000)
    arg_parser.add_argument("--tasks", nargs="+", default="all",
                            choices=["all", "simulate_chains",
                                     "plot_parameters", "get_r_squared"])
    arg_parser.add_argument("--show_progress", default=False,
                            action="store_true")
    arg_parser.add_argument("--integrator", type=str, default="dopri5")
    arg_parser.add_argument("--integrator_method", type=str, default="adams")

    return arg_parser.parse_args()

if __name__ == "__main__":
    main()
