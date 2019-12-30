#!/usr/bin/env python
import sys
import argparse
import numpy as np
from stan_helpers import StanSession, moving_average, get_prior_from_sample_file

def get_args():
    """parse command line arguments"""
    arg_parser = argparse.ArgumentParser(description="Infer parameters of " +
                                         "calcium mode using Stan.")
    arg_parser.add_argument("--stan_model", dest="stan_model", metavar="MODEL",
                            type=str, required=True)
    arg_parser.add_argument("--cell_id", dest="cell_id", metavar="N", type=int,
                            default=0)
    arg_parser.add_argument("--filter_type", dest="filter_type",
                            choices=["none", "moving_average"], default="none")
    arg_parser.add_argument("--moving_average_window",
                            dest="moving_average_window", type=int, default=20)
    arg_parser.add_argument("--t0", dest="t0", metavar="T", type=int,
                            default=200)
    arg_parser.add_argument("--num_chains", dest="num_chains", type=int,
                            default=4)
    arg_parser.add_argument("--num_iters", dest="num_iters", type=int,
                            default=3000)
    arg_parser.add_argument("--warmup", dest="warmup", type=int, default=1000)
    arg_parser.add_argument("--thin", dest="thin", type=int, default=1)
    arg_parser.add_argument("--result_dir", dest="result_dir", metavar="DIR",
                            type=str, default=".")
    arg_parser.add_argument("--prior_sample_file", dest="prior_sample_file",
                            metavar="FILE", type=str, default=None)
    arg_parser.add_argument("--prior_std_scale", dest="prior_std_scale",
                            type=float, default=1.0)

    return arg_parser.parse_args()

def main():
    # get command-line arguments
    args = get_args()
    stan_model = args.stan_model
    cell_id = args.cell_id
    filter_type = args.filter_type
    moving_average_window = args.moving_average_window
    t0 = args.t0
    num_chains = args.num_chains
    num_iters = args.num_iters
    warmup = args.warmup
    thin = args.thin
    result_dir = args.result_dir
    prior_sample_file = args.prior_sample_file
    prior_std_scale = args.prior_std_scale

    # prepare data for Stan model
    # get trajectory and time
    y_raw = np.loadtxt("canorm_tracjectories.csv", delimiter=",")
    t_end = 1000
    y = y_raw[cell_id, :]
    # apply preprocessing to the trajectory if specified
    if filter_type == "moving_average":
        print("Filtering raw trajectory using moving average with window " +
              "size of {}...".format(moving_average_window))
        y = moving_average(y, window=moving_average_window)
        y = np.squeeze(y)
    y0 = np.array([0, 0, 0.7, y[t0]])
    y = y[t0 + 1:]
    T = y.size
    ts = np.linspace(t0 + 1, t_end, t_end - t0)

    # get prior distribution
    if prior_sample_file is None:
        # no sample file provided. use Gaussian(1.0, 1.0) for all parameters
        num_params = 19
        prior_mean = np.ones(num_params)
        prior_std = np.ones(num_params)
    else:
        prior_mean, prior_std = get_prior_from_sample_file(prior_sample_file)

    prior_std *= prior_std_scale

    # gather prepared data
    calcium_data = {
        "N": 4,
        "T": T,
        "y0": y0,
        "y": y,
        "t0": t0,
        "ts": ts,
        "mu_prior": prior_mean,
        "sigma_prior": prior_std
    }
    print("Data initialized")
    sys.stdout.flush()

    # run Stan session
    stan_session = StanSession(stan_model, calcium_data, result_dir,
                               num_chains=num_chains, num_iters=num_iters,
                               warmup=warmup, thin=thin)
    stan_session.run_sampling()

if __name__ == "__main__":
    main()
