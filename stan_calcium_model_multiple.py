#!/usr/bin/env python
import sys
import os
import argparse
import numpy as np
import pandas as pd
from stan_helpers import StanSession, StanSampleAnalyzer, moving_average, \
    get_prior_from_sample_files, calcium_ode

def main():
    # get command-line arguments
    args = get_args()
    stan_model = args.stan_model
    cell_list = args.cell_list
    num_cells = args.num_cells
    filter_type = args.filter_type
    moving_average_window = args.moving_average_window
    t0 = args.t0
    num_chains = args.num_chains
    num_iters = args.num_iters
    warmup = args.warmup
    thin = args.thin
    result_dir = args.result_dir
    prior_std_scale = args.prior_std_scale

    # prepare data for Stan model
    # get trajectory and time
    print("Loading trajectories...")
    y = np.loadtxt("canorm_tracjectories.csv", delimiter=",")
    t_end = 1000
    # apply preprocessing to the trajectory if specified
    if filter_type == "moving_average":
        y = moving_average(y, window=moving_average_window)
    ts = np.linspace(t0 + 1, t_end, t_end - t0)
    T = ts.size - 1
    param_names = ["sigma", "KonATP", "L", "Katp", "KoffPLC", "Vplc", "Kip3",
                   "KoffIP3", "a", "dinh", "Ke", "Be", "d1", "d5", "epr",
                   "eta1", "eta2", "eta3", "c0", "k3"]

    cells = pd.read_csv(cell_list, index_col=False)
    cell_order = 0
    is_r_hat_good = True
    while cell_order < num_cells:
        if is_r_hat_good:
            # good R_hat, advance to the next cell
            cell_order += 1
            print("Initializing sampling for {}-th cell".format(cell_order))
        else:
            # bad R_hat, restart sampling for the cell
            print("Bad R_hat value of log probability for "
                  + "{}-th cell. ".format(cell_order))
            print("Re-initializing sampling...")

        # get cell and its predecessor
        cell_id = cells.loc[cell_order, "Cell"]
        cell_dir = os.path.join(result_dir, "cell-{:4d}".format(cell_id))
        if not os.path.exists(cell_dir):
            os.mkdir(cell_dir)
        pred_id = cells.loc[cell_order, "Parent"]
        pred_dir = os.path.join(result_dir, "cell-{:4d}".format(pred_id))

        # get prior distribution of predecessor
        pred_chains = [2] if pred_id == 0 else [0, 1, 2, 3]
        prior_mean, prior_std = get_prior_from_sample_files(pred_dir,
                                                            pred_chains)

        if prior_std_scale != 1.0:
            print("Scaling standard deviation of prior distribution by "
                  + "{}...".format(prior_std_scale))
            prior_std *= prior_std_scale

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
        sys.stdout.flush()

        # run Stan session
        stan_session = StanSession(stan_model, calcium_data, cell_dir,
                                   num_chains=num_chains, num_iters=num_iters,
                                   warmup=warmup, thin=thin)
        stan_session.run_sampling()
        log_prob_r_hat = stan_session.run_post_sampling_routines()

        # analyze result of current cell
        is_r_hat_good = 0.9 <= log_prob_r_hat <= 1.1
        if is_r_hat_good:
            analyzer = StanSampleAnalyzer(cell_dir, calcium_ode, ts, y0, 3,
                                          use_summary=True, y_ref=y_ref)
            analyzer.simulate_chains()
            analyzer.plot_parameters()
            analyzer.get_r_squared()

def get_args():
    """Parse command line arguments"""
    arg_parser = argparse.ArgumentParser(
        description="Infer parameters of calclium model for multiple cells "
                    + "using stan")
    arg_parser.add_argument("--stan_model", dest="stan_model", metavar="MODEL",
                            type=str, required=True)
    arg_parser.add_argument("--cell_list", dest="cell_list", type=str,
                            required=True)
    arg_parser.add_argument("--num_cells", dest="num_cells", type=int,
                            default=100)
    arg_parser.add_argument("--filter_type", dest="filter_type",
                            choices=["none", "moving_average"], default="none")
    arg_parser.add_argument("--t0", dest="t0", metavar="T", type=int,
                            default=200)
    arg_parser.add_argument("--num_chains", dest="num_chains", type=int,
                            default=4)
    arg_parser.add_argument("--num_iters", dest="num_iters", type=int,
                            default=2000)
    arg_parser.add_argument("--warmup", dest="warmup", type=int, default=1000)
    arg_parser.add_argument("--thin", dest="thin", type=int, default=1)
    arg_parser.add_argument("--result_dir", dest="result_dir", metavar="DIR",
                            type=str, default=".")
    arg_parser.add_argument("--prior_std_scale", dest="prior_std_scale",
                            type=float, default=1.0)

    return arg_parser.parse_args()

if __name__ == "__main__":
    main()