#!/usr/bin/env python
import sys
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
    ode_variant = args.ode_variant
    stan_backend = args.stan_backend
    cell_id = args.cell_id
    filter_type = args.filter_type
    moving_average_window = args.moving_average_window
    t0 = args.t0
    downsample_offset = args.downsample_offset
    prior_dir = args.prior_dir
    prior_chains = args.prior_chains
    prior_spec_path = args.prior_spec
    prior_std_scale = args.prior_std_scale
    num_chains = args.num_chains
    num_iters = args.num_iters
    warmup = args.warmup
    thin = args.thin
    adapt_delta = args.adapt_delta
    max_treedepth = args.max_treedepth
    result_dir = args.result_dir
    analysis_tasks = args.analysis_tasks
    use_summary = args.use_summary

    # prepare data for Stan model
    print("Initializing data for cell {}...".format(cell_id))
    # get trajectory and time
    y, y0_ca, ts = load_trajectories(
        t0, filter_type=filter_type,
        moving_average_window=moving_average_window,
        downsample_offset=downsample_offset)
    y = y[cell_id, :]
    y0_ca = y0_ca[cell_id]
    param_names = ["sigma", "KonATP", "L", "Katp", "KoffPLC", "Vplc", "Kip3",
                   "KoffIP3", "a", "dinh", "Ke", "Be", "d1", "d5", "epr",
                   "eta1", "eta2", "eta3", "c0", "k3"]
    var_names = ["PLC", "IP3", "h", "Ca"]
    calcium_ode = getattr(stan_helpers, "calcium_ode_" + ode_variant)

    # get prior distribution
    if prior_dir:
        prior_mean, prior_std = get_prior_from_sample_files(
            prior_dir, prior_chains)
    elif prior_spec_path:
        print(f"Getting prior distribution from {prior_spec_path}...")
        prior_spec = pd.read_csv(prior_spec_path, delimiter="\t", index_col=0)
        prior_mean = prior_spec["mu"].to_numpy()
        prior_std = prior_spec["sigma"].to_numpy()
        print("Prior distbution is set as follows:")
        print(prior_spec)
    else:
        # no sample file provided. use Gaussian(1.0, 1.0) for all parameters
        print("Setting prior distribution to N(1.0, 1.0) for all parameters")
        num_params = len(param_names) - 1
        prior_mean = np.ones(num_params)
        prior_std = np.ones(num_params)

    if prior_std_scale != 1.0 and not prior_spec_path:
        print("Scaling standard deviation of prior distribution by "
              + "{}...".format(prior_std_scale))
        prior_std *= prior_std_scale

    # gather prepared data
    # y0 = np.array([0, 0, 0.7, y[t0]])
    # y_ref = [None, None, None, y[t0 + 1:]]
    y0 = np.array([0, 0, 0.7, y0_ca])
    y_ref = [None, None, None, y]
    T = ts.size
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
    print("Data initialized")
    sys.stdout.flush()

    # run Stan session
    control = {"adapt_delta": adapt_delta, "max_treedepth": max_treedepth}
    print("The following NUTS parameters will be used:")
    print(control)
    sys.stdout.flush()
    stan_session = StanSession(stan_model, result_dir,
                               stan_backend=stan_backend, data=calcium_data,
                               num_chains=num_chains, num_iters=num_iters,
                               warmup=warmup, thin=thin)
    stan_session.run_sampling(control=control)
    stan_session.gather_fit_result()

    # run analysis on Stan results
    if analysis_tasks:
        analyzer = StanSessionAnalyzer(result_dir, use_summary=use_summary,
                                       param_names=param_names)
        if "all" in analysis_tasks:
            analysis_tasks = ["simulate_chains", "plot_parameters",
                              "get_r_squared"]
        if "simulate_chains" in analysis_tasks:
            analyzer.simulate_chains(calcium_ode, t0, ts, y0, y_ref=y_ref,
                                     var_names=var_names)
        if "plot_parameters" in analysis_tasks:
            analyzer.plot_parameters()
        if "get_r_squared" in analysis_tasks:
            analyzer.get_r_squared()

def get_args():
    """parse command line arguments"""
    arg_parser = argparse.ArgumentParser(
        description="Infer parameters of calcium mode using stan.")
    arg_parser.add_argument("--stan_model", metavar="MODEL", type=str,
                            required=True)
    arg_parser.add_argument("--ode_variant", type=str, default="vanilla",
                            choices=["vanilla", "equiv_1", "equiv_2",
                                     "const_1", "const_2"])
    arg_parser.add_argument("--stan_backend",  metavar="BACKEND", type=str,
                            default="pystan", choices=["pystan", "cmdstanpy"])
    arg_parser.add_argument("--cell_id", type=int, default=0)
    arg_parser.add_argument("--filter_type", default=None,
                            choices=["moving_average"])
    arg_parser.add_argument("--moving_average_window", type=int, default=20)
    arg_parser.add_argument("--t0", type=int, default=200)
    arg_parser.add_argument("--downsample_offset", type=int, default=300)
    arg_parser.add_argument("--prior_dir", type=str, default=None)
    arg_parser.add_argument("--prior_chains", type=int, nargs="+",
                            default=[0, 1, 2, 3])
    arg_parser.add_argument("--prior_spec", type=str, default=None)
    arg_parser.add_argument("--prior_std_scale", type=float, default=1.0)
    arg_parser.add_argument("--num_chains", type=int, default=4)
    arg_parser.add_argument("--num_iters", type=int, default=2000)
    arg_parser.add_argument("--warmup", type=int, default=1000)
    arg_parser.add_argument("--thin", type=int, default=1)
    arg_parser.add_argument("--adapt_delta",  type=float, default=0.8)
    arg_parser.add_argument("--max_treedepth", type=int, default=10)
    arg_parser.add_argument("--result_dir", type=str, required=True)
    arg_parser.add_argument("--analysis_tasks", nargs="+", default=None,
                            choices=["all", "simulate_chains",
                                     "plot_parameters", "get_r_squared"])
    arg_parser.add_argument("--use_summary", default=False,
                            action="store_true")

    return arg_parser.parse_args()

if __name__ == "__main__":
    main()
