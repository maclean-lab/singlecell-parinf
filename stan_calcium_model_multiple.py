#!/usr/bin/env python
import sys
import os
import argparse
import numpy as np
import pandas as pd
import stan_helpers
from stan_helpers import StanSession, StanSessionAnalyzer, load_trajectories, \
    get_prior_from_sample_files

def main():
    # get command-line arguments
    args = get_args()
    stan_model = args.stan_model
    cell_list_path = args.cell_list
    num_cells = args.num_cells
    filter_type = args.filter_type
    moving_average_window = args.moving_average_window
    var_mask = args.var_mask
    param_mask = args.param_mask
    ode_variant = args.ode_variant
    t0 = args.t0
    downsample_offset = args.downsample_offset
    prior_id = args.prior_cell
    prior_chains = args.prior_chains
    prior_clip_min = args.prior_clip_min
    prior_clip_max = args.prior_clip_max
    prior_std_scale = args.prior_std_scale
    num_chains = args.num_chains
    num_iters = args.num_iters
    warmup = args.warmup
    thin = args.thin
    adapt_delta = args.adapt_delta
    max_treedepth = args.max_treedepth
    rhat_upper_bound = args.rhat_upper_bound
    result_dir = args.result_dir

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
    prior_dir = os.path.join(result_dir, f"cell-{prior_id:04d}")

    # get trajectory and time
    y, y0_ca, ts = load_trajectories(
        t0, filter_type=filter_type,
        moving_average_window=moving_average_window,
        downsample_offset=downsample_offset, verbose=True)
    T = ts.size
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

    calcium_ode = getattr(stan_helpers, "calcium_ode_" + ode_variant)
    control = {"adapt_delta": adapt_delta, "max_treedepth": max_treedepth}

    max_num_tries = 3  # maximum number of tries of stan sampling
    # convert 1, 2, 3... to 1st, 2nd, 3rd...
    # credit: https://stackoverflow.com/questions/9647202
    num2ord = lambda n: \
        "{}{}".format(n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
    for cell_order in range(first_cell_order, last_cell_order + 1):
        # set up for current cell
        cell_id = cell_list.loc[cell_order, "Cell"]
        cell_dir = os.path.join(result_dir, f"cell-{cell_id:04d}")
        if not os.path.exists(cell_dir):
            os.mkdir(cell_dir)
        num_tries = 0
        print(f"Initializing sampling for the {num2ord(cell_order)} cell "
              + f"(ID: {cell_id})...")

        # update prior distribution
        if prior_chains:
            print("Updating prior distribution from the "
                  + f"{num2ord(cell_order - 1)} cell (ID: {prior_id})...")
            prior_mean, prior_std = get_prior_from_sample_files(prior_dir,
                                                                prior_chains)

            if prior_std_scale != 1.0:
                print("Scaling standard deviation of prior distribution "
                      + f"by {prior_std_scale}...")
                prior_std *= prior_std_scale

            # restrict standard deviation of prior
            prior_std = np.clip(prior_std, prior_clip_min, prior_clip_max)

            prior_chains = None  # reset prior chains
        else:
            print("Prior distribution will not be updated from the "
                  + f"{num2ord(cell_order - 1)} cell (ID: {prior_id}) "
                  + "due to too many unsuccessful attempts of sampling")

        # gather prepared data
        y0 = np.array([0, 0, 0.7, y0_ca[cell_id]])
        y_ref = [None, None, None, y[cell_id, :]]
        if var_mask:
            y0 = np.array(
                [y0[i] for i, mask in enumerate(var_mask) if mask == "1"])
            y_ref = [y_ref[i] for i, mask in enumerate(var_mask) if mask == "1"]
        calcium_data = {
            "N": 4,
            "T": T,
            "y0": y0,
            "y": y_ref[-1],
            "t0": 0,
            "ts": ts,
            "mu_prior": prior_mean,
            "sigma_prior": prior_std
        }
        sys.stdout.flush()

        # try sampling
        while not prior_chains and num_tries < max_num_tries:
            num_tries += 1
            print(f"Starting {num2ord(num_tries)} attempt of sampling...")
            sys.stdout.flush()

            # run Stan session
            stan_session = StanSession(
                stan_model, cell_dir, data=calcium_data,
                num_chains=num_chains, num_iters=num_iters, warmup=warmup,
                thin=thin, rhat_upper_bound=rhat_upper_bound)
            stan_session.run_sampling(control=control)
            stan_session.gather_fit_result()

            # find chain combo with good R_hat value
            prior_chains = stan_session.get_good_chain_combo()

            if prior_chains:
                # good R_hat value of one chain combo
                # analyze result of current cell
                print("Good R_hat value of log posteriors for the "
                      + f"{num2ord(cell_order)} cell (ID: {cell_id})")
                print("Running analysis on sampled result...")

                analyzer = StanSessionAnalyzer(
                    cell_dir, use_summary=True, param_names=param_names)
                analyzer.simulate_chains(calcium_ode, 0, ts, y0, y_ref=y_ref,
                                         var_names=var_names)
                analyzer.plot_parameters()
                analyzer.get_r_squared()
            else:
                # bad R_hat value of every chain combo
                print("Bad R_hat value of log posteriors for the "
                      + f"{num2ord(cell_order)} cell (ID: {cell_id}) on the "
                      + f"{num2ord(num_tries)} attempt")

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
                    + "using Stan")
    arg_parser.add_argument("--stan_model", metavar="MODEL", type=str,
                            required=True)
    arg_parser.add_argument("--ode_variant", type=str, default="vanilla",
                            choices=["vanilla", "equiv_1", "equiv_2",
                                     "const_1", "const_2"])
    arg_parser.add_argument("--cell_list", type=str, required=True)
    arg_parser.add_argument("--num_cells", type=int, default=100)
    arg_parser.add_argument("--filter_type", default=None,
                            choices=["moving_average"])
    arg_parser.add_argument("--moving_average_window", type=int, default=20)
    arg_parser.add_argument("--t0", type=int, default=200)
    arg_parser.add_argument("--downsample_offset", type=int, default=300)
    arg_parser.add_argument("--prior_cell", type=int, default=0)
    arg_parser.add_argument("--prior_chains", type=int, nargs="+",
                            default=[2])
    arg_parser.add_argument("--prior_std_scale",  type=float, default=1.0)
    arg_parser.add_argument("--prior_clip_min",  type=float, default=0.001)
    arg_parser.add_argument("--prior_clip_max",  type=float, default=5)
    arg_parser.add_argument("--num_chains", type=int, default=4)
    arg_parser.add_argument("--num_iters", type=int, default=2000)
    arg_parser.add_argument("--warmup", type=int, default=1000)
    arg_parser.add_argument("--thin", type=int, default=1)
    arg_parser.add_argument("--adapt_delta", type=float, default=0.8)
    arg_parser.add_argument("--max_treedepth", type=int, default=10)
    arg_parser.add_argument("--rhat_upper_bound", type=float, default=1.1)
    arg_parser.add_argument("--result_dir", type=str, required=True)

    return arg_parser.parse_args()

if __name__ == "__main__":
    main()
