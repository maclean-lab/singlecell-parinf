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
    T = ts.size
    param_names = ["sigma", "KonATP", "L", "Katp", "KoffPLC", "Vplc", "Kip3",
                   "KoffIP3", "a", "dinh", "Ke", "Be", "d1", "d5", "epr",
                   "eta1", "eta2", "eta3", "c0", "k3"]

    cells = pd.read_csv(cell_list, delimiter="\t", index_col=False)
    cell_order = 0  # order of cell in cell list, not cell id
    cell_id = cells.loc[cell_order, "Cell"]
    cell_dir = os.path.join(result_dir, "cell-{:04d}".format(cell_id))
    is_r_hat_good = True
    num_runs = 0
    max_num_runs = 3
    # convert 1, 2, 3... to 1st, 2nd, 3rd...
    # credit: https://stackoverflow.com/questions/9647202
    num2ord = lambda n: \
        "{}{}".format(n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
    while cell_order < num_cells:
        # prepare for current iteration
        if is_r_hat_good or num_runs == max_num_runs:
            # good R_hat or too many runs, advance to the next cell
            if num_runs == max_num_runs:
                # too many runs, skip last cell
                print("Skip the {} cell (ID: {}) ".format(num2ord(cell_order),
                                                          cell_id)
                      + "due to too many runs of sampling")
                print("Prior distribution will not be updated")
            else:
                # update prior distribution
                print("Getting prior distribution from the "
                      + "{} cell (ID: {})".format(num2ord(cell_order), cell_id))
                prior_chains = [2] if cell_id == 0 else [0, 1, 2, 3]
                prior_mean, prior_std = get_prior_from_sample_files(
                    cell_dir, prior_chains
                )

                if prior_std_scale != 1.0:
                    print("Scaling standard deviation of prior distribution "
                          + "by {}...".format(prior_std_scale))
                    prior_std *= prior_std_scale

            # get current cell
            cell_order += 1
            cell_id = cells.loc[cell_order, "Cell"]
            cell_dir = os.path.join(result_dir, "cell-{:04d}".format(cell_id))
            if not os.path.exists(cell_dir):
                os.mkdir(cell_dir)
            num_runs = 1

            # gather prepared data
            print("Initializing sampling for the "
                  + "{} cell (ID: {})...".format(num2ord(cell_order), cell_id))
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
            print("Sampling initialized. Starting sampling...")
        else:
            # bad R_hat, restart sampling for the cell
            print("Bad R_hat value of log probability for the "
                  + "{} cell (ID: {})".format(num2ord(cell_order), cell_id))
            print("Restarting sampling...")

            num_runs += 1

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
            print("Good R_hat value of log probability the "
                + "{} cell (ID: {})".format(num2ord(cell_order), cell_id))

            analyzer = StanSampleAnalyzer(cell_dir, calcium_ode, ts, y0, 3,
                                          use_summary=True,
                                          param_names=param_names, y_ref=y_ref)
            analyzer.simulate_chains()
            analyzer.plot_parameters()
            analyzer.get_r_squared()

        print()
        sys.stdout.flush()

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
    arg_parser.add_argument("--moving_average_window",
                            dest="moving_average_window", type=int, default=20)
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
