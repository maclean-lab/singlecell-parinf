#!/usr/bin/env python
import os
import sys
import pickle
import argparse
import numpy as np
import scipy.integrate
import pandas as pd
import matplotlib.pyplot as plt
from stan_helpers import StanSession, StanSessionAnalyzer, moving_average, \
    calcium_ode_vanilla

def main():
    # get command-line arguments
    args = get_args()
    stan_model = args.stan_model
    filter_type = args.filter_type
    moving_average_window = args.moving_average_window
    t0 = args.t0
    num_cells = args.num_cells
    result_dir = args.result_dir
    prior_std_scale = args.prior_std_scale

    # prepare data for Stan model
    # get trajectory and time
    print("Loading trajectories...")
    y = np.loadtxt("canorm_trajectories.csv", delimiter=",")
    if not num_cells:
        num_cells = y.shape[0]
    t_end = y.shape[1] - 1
    # apply preprocessing to the trajectory if specified
    if filter_type == "moving_average":
        y = moving_average(y, window=moving_average_window)
        y = np.squeeze(y)
    ts = np.linspace(t0 + 1, t_end, t_end - t0)
    # downsample trajectories
    t_downsample = 300
    y = np.concatenate((y[:, 0:t_downsample], y[:, t_downsample::10]), axis=1)
    ts = np.concatenate((ts[0:t_downsample-t0], ts[t_downsample-t0::10]))
    T = ts.size
    # y_sim = np.zeros((num_cells, T))
    param_names = ["sigma", "KonATP", "L", "Katp", "KoffPLC", "Vplc", "Kip3",
                   "KoffIP3", "a", "dinh", "Ke", "Be", "d1", "d5", "epr",
                   "eta1", "eta2", "eta3", "c0", "k3"]

    print("Setting prior distribution to N(1.0, 1.0) for all parameters")
    num_params = len(param_names) - 1
    prior_mean = np.ones(num_params)
    prior_std = np.ones(num_params)
    if prior_std_scale != 1.0:
        print("Scaling standard deviation of prior distribution by "
              + "{}...".format(prior_std_scale))
        prior_std *= prior_std_scale

    # get stan model
    stan_session = StanSession(stan_model, result_dir)

    for cell_id in range(num_cells):
        print(f"Initializing data for cell {cell_id}")
        cell_dir = os.path.join(result_dir, f"cell-{cell_id:04d}")
        if not os.path.exists(cell_dir):
            os.mkdir(cell_dir)

        # gather prepared data
        y0 = np.array([0, 0, 0.7, y[cell_id, t0]])
        y_ref = y[cell_id, t0 + 1:]
        calcium_data = {
            "N": 4,
            "T": T,
            "y0": y0,
            "y": y_ref,
            "t0": t0,
            "ts": ts,
            "mu_prior": prior_mean,
            "sigma_prior": prior_std
        }
        print("Data initialized")

        # run Stan optimization
        print(f"Running variational Bayes for cell {cell_id}")
        sys.stdout.flush()

        _ = stan_session.model.vb(
            data=calcium_data,
            sample_file=os.path.join(cell_dir, "chain_0.csv"),
            diagnostic_file=os.path.join(cell_dir, "vb_diagnostic"))

        analyzer = StanSessionAnalyzer(cell_dir, stan_operation="vb",
                                       use_fit_export=False, num_chains=1,
                                       warmup=0, param_names=param_names)
        _ = analyzer.simulate_chains(calcium_ode_vanilla, t0, ts, y0,
                                     y_ref=y_ref)

def get_args():
    """parse command line arguments"""
    arg_parser = argparse.ArgumentParser(
        description="Infer parameters of calcium mode using stan.")
    arg_parser.add_argument("--stan_model", dest="stan_model", metavar="MODEL",
                            type=str, required=True)
    arg_parser.add_argument("--filter_type", dest="filter_type",
                            choices=["none", "moving_average"], default="none")
    arg_parser.add_argument("--moving_average_window",
                            dest="moving_average_window", type=int, default=20)
    arg_parser.add_argument("--t0", dest="t0", metavar="T", type=int,
                            default=200)
    arg_parser.add_argument("--num_cells", dest="num_cells", type=int,
                            default=0)
    arg_parser.add_argument("--result_dir", dest="result_dir", metavar="DIR",
                            type=str, default=".")
    arg_parser.add_argument("--prior_std_scale", dest="prior_std_scale",
                            type=float, default=1.0)

    return arg_parser.parse_args()

if __name__ == "__main__":
    main()
