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
    rhat_upper_bound = args.rhat_upper_bound
    result_dir = args.result_dir
    prior_std_scale = args.prior_std_scale

    # prepare data for Stan model
    # get trajectory and time
    print("Loading trajectories...")
    y = np.loadtxt("canorm_tracjectories.csv", delimiter=",")
    t_end = y.shape[1]
    # apply preprocessing to the trajectory if specified
    if filter_type == "moving_average":
        y = moving_average(y, window=moving_average_window)
    ts = np.linspace(t0 + 1, t_end, t_end - t0)
    # downsample trajectories
    y = np.concatenate((y[:, 0:400], y[:, 400::10]), axis=1)
    ts = np.concatenate((ts[0:400-t0], ts[400-t0::10]))
    T = ts.size
    param_names = ["sigma", "KonATP", "L", "Katp", "KoffPLC", "Vplc", "Kip3",
                   "KoffIP3", "a", "dinh", "Ke", "Be", "d1", "d5", "epr",
                   "eta1", "eta2", "eta3", "c0", "k3"]

    cells = pd.read_csv(cell_list, delimiter="\t", index_col=False)
    prev_id = 0
    prior_dir = os.path.join(result_dir, "cell-0000")
    prior_chains = [2]  # chains for prior (for cell 0, use chain 2 only)
    max_num_tries = 3  # maximum number of tries of stan sampling
    # convert 1, 2, 3... to 1st, 2nd, 3rd...
    # credit: https://stackoverflow.com/questions/9647202
    num2ord = lambda n: \
        "{}{}".format(n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
    for cell_order in range(1, num_cells + 1):
        # set up for current cell
        cell_id = cells.loc[cell_order, "Cell"]
        cell_dir = os.path.join(result_dir, "cell-{:04d}".format(cell_id))
        if not os.path.exists(cell_dir):
            os.mkdir(cell_dir)
        num_tries = 0
        print("Initializing sampling for the "
              + "{} cell (ID: {})...".format(num2ord(cell_order), cell_id))

        # update prior distribution
        if prior_chains:
            print("Updating prior distribution from the "
                  + "{} cell (ID: ".format(num2ord(cell_order - 1))
                  + "{})...".format(prev_id))
            prior_mean, prior_std = get_prior_from_sample_files(prior_dir,
                                                                prior_chains)

            if prior_std_scale != 1.0:
                print("Scaling standard deviation of prior distribution "
                      + "by {}...".format(prior_std_scale))
                prior_std *= prior_std_scale

            prior_chains = None  # reset prior chains
        else:
            print("Prior distribution will not be update from the "
                  + "{} cell (ID: ".format(num2ord(cell_order - 1))
                  + "{}) due to too many unsuccessful ".format(prev_id)
                  + "attempts of sampling")

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
        sys.stdout.flush()

        # try sampling
        while not prior_chains and num_tries < max_num_tries:
            num_tries += 1
            print("Starting {} attempt of ".format(num2ord(num_tries))
                  + "sampling...")
            sys.stdout.flush()

            # run Stan session
            stan_session = StanSession(stan_model, calcium_data, cell_dir,
                                    num_chains=num_chains, num_iters=num_iters,
                                    warmup=warmup, thin=thin,
                                    rhat_upper_bound=rhat_upper_bound)
            stan_session.run_sampling()
            _ = stan_session.gather_fit_result()

            # find chain combo with good R_hat value
            prior_chains = stan_session.get_good_chain_combo()

            if prior_chains:
                # good R_hat value of one chain combo
                # analyze result of current cell
                print("Good R_hat value of log posteriors for the "
                      + "{} cell (ID: {})".format(num2ord(cell_order), cell_id))
                print("Running analysis on sampled result...")

                analyzer = StanSampleAnalyzer(cell_dir, calcium_ode, 3, y0, t0,
                                              ts, use_summary=True,
                                              param_names=param_names,
                                              y_ref=y_ref)
                analyzer.simulate_chains()
                analyzer.plot_parameters()
                analyzer.get_r_squared()
            else:
                # bad R_hat value of every chain combo
                print("Bad R_hat value of log posteriors for the "
                      + "{} cell (ID: {})".format(num2ord(cell_order), cell_id)
                      + "on the {} attempt".format(num2ord(num_tries)))

            print()
            sys.stdout.flush()

        # prepare for next cell
        prev_id = cell_id
        prior_dir = cell_dir

    print("Sampling for all cells finished")
    sys.stdout.flush()

def get_args():
    """Parse command line arguments"""
    arg_parser = argparse.ArgumentParser(
        description="Infer parameters of calclium model for multiple cells "
                    + "using stan"
    )
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
    arg_parser.add_argument("--rhat_upper_bound", dest="rhat_upper_bound",
                            type=float, default=1.1)
    arg_parser.add_argument("--result_dir", dest="result_dir", metavar="DIR",
                            type=str, default=".")
    arg_parser.add_argument("--prior_std_scale", dest="prior_std_scale",
                            type=float, default=1.0)

    return arg_parser.parse_args()

if __name__ == "__main__":
    main()
