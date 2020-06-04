#!/usr/bin/env python
import sys
import os
import argparse
import numpy as np
import pandas as pd
from stan_helpers import StanSession, StanSessionAnalyzer, moving_average, \
    get_prior_from_sample_files, calcium_ode_vanilla, calcium_ode_equiv

def main():
    # get command-line arguments
    args = get_args()
    stan_model = args.stan_model
    ode_variant = args.ode_variant
    cell_list_path = args.cell_list
    prior_id = args.prior_cell
    prior_chains = args.prior_chains
    num_cells = args.num_cells
    filter_type = args.filter_type
    moving_average_window = args.moving_average_window
    t0 = args.t0
    num_chains = args.num_chains
    num_iters = args.num_iters
    warmup = args.warmup
    thin = args.thin
    adapt_delta = args.adapt_delta
    max_treedepth = args.max_treedepth
    rhat_upper_bound = args.rhat_upper_bound
    result_dir = args.result_dir
    prior_std_scale = args.prior_std_scale

    # set up for sampling
    cell_list = pd.read_csv(cell_list_path, delimiter="\t", index_col=False)
    prior_cell_order = np.where(cell_list["Cell"] == prior_id)[0][0]
    num_total_cells = cell_list.shape[0]
    num_cells_left = num_total_cells - prior_cell_order - 1
    if num_cells == 0 or num_cells_left == 0:
        print("No cells to sample. Exiting...")
        sys.exit(0)
    if num_cells_left < num_cells:
        num_cells = num_cells_left
        print("The given number of cells exceeds the number of cells after "
              + "the prior cell in the cell list. Setting number of cells "
              + "to the latter")
    first_cell_order = prior_cell_order + 1
    last_cell_order = prior_cell_order + num_cells
    prior_dir = os.path.join(result_dir, "cell-{:04d}".format(prior_id))

    # get trajectory and time
    print("Loading trajectories...")
    y = np.loadtxt("canorm_tracjectories.csv", delimiter=",")
    t_end = y.shape[1] - 1
    # apply preprocessing to the trajectory if specified
    if filter_type == "moving_average":
        y = moving_average(y, window=moving_average_window)
    ts = np.linspace(t0 + 1, t_end, t_end - t0)
    # downsample trajectories
    t_downsample = 300
    y = np.concatenate((y[:, 0:t_downsample], y[:, t_downsample::10]), axis=1)
    ts = np.concatenate((ts[0:t_downsample-t0], ts[t_downsample-t0::10]))
    T = ts.size
    param_names = ["sigma", "KonATP", "L", "Katp", "KoffPLC", "Vplc", "Kip3",
                   "KoffIP3", "a", "dinh", "Ke", "Be", "d1", "d5", "epr",
                   "eta1", "eta2", "eta3", "c0", "k3"]
    var_names = ["PLC", "IP3", "h", "Ca"]
    if ode_variant == "equiv":
        calcium_ode = calcium_ode_equiv
    else:
        calcium_ode = calcium_ode_vanilla
    control = {"adapt_delta": adapt_delta, "max_treedepth": max_treedepth}

    max_num_tries = 3  # maximum number of tries of stan sampling
    # convert 1, 2, 3... to 1st, 2nd, 3rd...
    # credit: https://stackoverflow.com/questions/9647202
    num2ord = lambda n: \
        "{}{}".format(n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
    for cell_order in range(first_cell_order, last_cell_order + 1):
        # set up for current cell
        cell_id = cell_list.loc[cell_order, "Cell"]
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
                  + "{})...".format(prior_id))
            prior_mean, prior_std = get_prior_from_sample_files(prior_dir,
                                                                prior_chains)

            if prior_std_scale != 1.0:
                print("Scaling standard deviation of prior distribution "
                      + "by {}...".format(prior_std_scale))
                prior_std *= prior_std_scale

            prior_chains = None  # reset prior chains
        else:
            print("Prior distribution will not be updated from the "
                  + "{} cell (ID: ".format(num2ord(cell_order - 1))
                  + "{}) due to too many unsuccessful ".format(prior_id)
                  + "attempts of sampling")

        # gather prepared data
        y0 = np.array([0, 0, 0.7, y[cell_id, t0]])
        y_ref = [None, None, None, y[cell_id, t0 + 1:]]
        calcium_data = {
            "N": 4,
            "T": T,
            "y0": y0,
            "y": y_ref[3],
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
            stan_session = StanSession(stan_model, cell_dir, data=calcium_data,
                                       num_chains=num_chains,
                                       num_iters=num_iters, warmup=warmup,
                                       thin=thin,
                                       rhat_upper_bound=rhat_upper_bound)
            stan_session.run_sampling(control=control)
            stan_session.gather_fit_result()

            # find chain combo with good R_hat value
            prior_chains = stan_session.get_good_chain_combo()

            if prior_chains:
                # good R_hat value of one chain combo
                # analyze result of current cell
                print("Good R_hat value of log posteriors for the "
                      + "{} cell (ID: {})".format(num2ord(cell_order), cell_id))
                print("Running analysis on sampled result...")

                analyzer = StanSessionAnalyzer(
                    cell_dir, use_summary=True, param_names=param_names)
                analyzer.simulate_chains(calcium_ode, t0, ts, y0, y_ref=y_ref,
                                         var_names=var_names)
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
        prior_id = cell_id
        prior_dir = cell_dir

    print("Sampling for all cells finished")
    sys.stdout.flush()

def get_args():
    """Parse command line arguments"""
    arg_parser = argparse.ArgumentParser(
        description="Infer parameters of calclium model for multiple cells "
                    + "using Stan"
    )
    arg_parser.add_argument("--stan_model", dest="stan_model", metavar="MODEL",
                            type=str, required=True)
    arg_parser.add_argument("--ode_variant", dest="ode_variant", type=str,
                            default="original",
                            choices=["original", "equiv", "const"])
    arg_parser.add_argument("--cell_list", dest="cell_list", type=str,
                            required=True)
    arg_parser.add_argument("--prior_cell", dest="prior_cell", type=int,
                            default=0)
    arg_parser.add_argument("--prior_chains", dest = "prior_chains", type=int,
                            nargs="+", default=[2])
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
    arg_parser.add_argument("--adapt_delta", dest="adapt_delta", type=float,
                            default=0.8)
    arg_parser.add_argument("--max_treedepth", dest="max_treedepth", type=int,
                            default=10)
    arg_parser.add_argument("--rhat_upper_bound", dest="rhat_upper_bound",
                            type=float, default=1.1)
    arg_parser.add_argument("--result_dir", dest="result_dir", metavar="DIR",
                            type=str, default=".")
    arg_parser.add_argument("--prior_std_scale", dest="prior_std_scale",
                            type=float, default=1.0)

    return arg_parser.parse_args()

if __name__ == "__main__":
    main()
