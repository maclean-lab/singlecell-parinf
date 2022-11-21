#!/usr/bin/env python
import sys
import os
import argparse
import numpy as np
import pandas as pd
import calcium_models
from stan_helpers import StanSession, StanSessionAnalyzer, load_trajectories, \
    get_prior_from_samples

def main():
    # get command-line arguments
    args = get_args()
    stan_model = args.stan_model
    cell_id = args.cell_id
    num_cycles = args.num_cycles
    filter_type = args.filter_type
    moving_average_window = args.moving_average_window
    var_mask = args.var_mask
    param_mask = args.param_mask
    ode_variant = args.ode_variant
    t0 = args.t0
    downsample_offset = args.downsample_offset
    prior_spec_path = args.prior_spec
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

    # get trajectory and time
    y, y0_ca, ts = load_trajectories(
        t0, filter_type=filter_type,
        moving_average_window=moving_average_window,
        downsample_offset=downsample_offset, verbose=True)
    T = ts.size
    # filter parameters
    if param_mask:
        param_names = [calcium_models.param_names[i + 1]
                       for i, mask in enumerate(param_mask) if mask == '1']
        param_names = ['sigma'] + param_names

    var_names = ['PLC', 'IP3', 'h', 'Ca']
    # filter variables
    if var_mask:
        var_names = [var_names[i] for i, mask in enumerate(var_mask)
                     if mask == '1']

    calcium_ode = getattr(calcium_models, 'calcium_ode_' + ode_variant)
    control = {'adapt_delta': adapt_delta, 'max_treedepth': max_treedepth}

    # set up prior for round 0
    prior_chains = None
    if prior_spec_path:
        print(f'Getting prior distribution from {prior_spec_path}...')
        prior_spec = pd.read_csv(prior_spec_path, delimiter='\t', index_col=0)
        prior_mean = prior_spec['mu'].to_numpy()
        prior_std = prior_spec['sigma'].to_numpy()
        print('Prior distbution is set as follows:')
        print(prior_spec)
    else:
        # no sample file provided. use Gaussian(1.0, 1.0) for all parameters
        print('Setting prior distribution to N(1.0, 1.0) for all parameters')
        num_params = len(param_names) - 1
        prior_mean = np.ones(num_params)
        prior_std = np.ones(num_params)


    max_num_tries = 3  # maximum number of tries of stan sampling
    # convert 1, 2, 3... to 1st, 2nd, 3rd...
    # credit: https://stackoverflow.com/questions/9647202
    num2ord = lambda n: \
        '{}{}'.format(n,'tsnrhtdd'[(n//10%10!=1)*(n%10<4)*n%10::4])
    for cycle in range(num_cycles):
        # set up for current round
        curr_dir = os.path.join(result_dir, f'round-{cycle:02d}')
        if not os.path.exists(curr_dir):
            os.mkdir(curr_dir)
        num_tries = 0
        print(f'Initializing sampling for round {cycle}...')

        # update prior distribution
        if cycle == 0:
            print(f'Using Lemon prior from round {cycle}...')
        elif prior_chains:
            print(f'Updating prior distribution from round {cycle}...')
            prior_mean, prior_std = get_prior_from_samples(prior_dir,
                                                           prior_chains)

            if prior_std_scale != 1.0:
                print('Scaling standard deviation of prior distribution '
                      + f'by {prior_std_scale}...')
                prior_std *= prior_std_scale

            # restrict standard deviation of prior
            prior_std = np.clip(prior_std, prior_clip_min, prior_clip_max)

            prior_chains = None  # reset prior chains
        else:
            print('Prior distribution will not be updated due to too many '
                  + 'unsuccessful attempts of sampling')

        # gather prepared data
        y0 = np.array([0, 0, 0.7, y0_ca[cell_id]])
        y_ref = [None, None, None, y[cell_id, :]]
        if var_mask:
            y0 = np.array(
                [y0[i] for i, mask in enumerate(var_mask) if mask == '1'])
            y_ref = [y_ref[i] for i, mask in enumerate(var_mask) if mask == '1']
        calcium_data = {
            'N': 4,
            'T': T,
            'y0': y0,
            'y': y_ref[-1],
            't0': 0,
            'ts': ts,
            'mu_prior': prior_mean,
            'sigma_prior': prior_std
        }
        sys.stdout.flush()

        # try sampling
        while not prior_chains and num_tries < max_num_tries:
            num_tries += 1
            print(f'Starting {num2ord(num_tries)} attempt of sampling...',
                  flush=True)

            # run Stan session
            stan_session = StanSession(
                stan_model, curr_dir, data=calcium_data,
                num_chains=num_chains, num_iters=num_iters, warmup=warmup,
                thin=thin, control=control, rhat_upper_bound=rhat_upper_bound)
            stan_session.run_sampling()
            stan_session.gather_fit_result()

            # find chain combo with good R_hat value
            prior_chains = stan_session.get_mixed_chains()

            if prior_chains:
                # good R_hat value of one chain combo
                # analyze result of current cell
                print(f'Good R_hat value of log posteriors for round {cycle}')
                print(f'Mixed chains are {", ".join(map(str, prior_chains))}')
                print('Running analysis on sampled result...', flush=True)

                analyzer = StanSessionAnalyzer(
                    curr_dir, sample_source='arviz_inf_data',
                    param_names=param_names)
                _ = analyzer.simulate_chains(calcium_ode, 0, ts, y0,
                                             y_ref=y_ref, var_names=var_names)
                analyzer.plot_parameters()
                analyzer.get_r_squared()
            else:
                # bad R_hat value of every chain combo
                print(f'Bad R_hat value of log posteriors for round {cycle}',
                      flush=True)

            print('', flush=True)

        # prepare for next cell
        prior_dir = curr_dir

    print('Sampling finished', flush=True)

def get_args():
    """Parse command line arguments"""
    arg_parser = argparse.ArgumentParser(
        description='Infer parameters of calclium model for multiple cells '
                    + 'using Stan')
    arg_parser.add_argument('--stan_model', type=str, required=True)
    arg_parser.add_argument('--ode_variant', type=str, required=True)
    arg_parser.add_argument('--cell_id', type=int, required=True)
    arg_parser.add_argument('--num_cycles', type=int, default=100)
    arg_parser.add_argument('--filter_type', default=None,
                            choices=['moving_average'])
    arg_parser.add_argument('--moving_average_window', type=int, default=20)
    arg_parser.add_argument('--t0', type=int, default=200)
    arg_parser.add_argument('--downsample_offset', type=int, default=300)
    arg_parser.add_argument('--var_mask', type=str, default=None)
    arg_parser.add_argument('--param_mask', type=str, default=None)
    arg_parser.add_argument('--prior_spec', type=str, default=None)
    arg_parser.add_argument('--prior_std_scale', type=float, default=1.0)
    arg_parser.add_argument('--prior_clip_min', type=float, default=0.001)
    arg_parser.add_argument('--prior_clip_max', type=float, default=5)
    arg_parser.add_argument('--num_chains', type=int, default=4)
    arg_parser.add_argument('--num_iters', type=int, default=2000)
    arg_parser.add_argument('--warmup', type=int, default=1000)
    arg_parser.add_argument('--thin', type=int, default=1)
    arg_parser.add_argument('--adapt_delta', type=float, default=0.8)
    arg_parser.add_argument('--max_treedepth', type=int, default=10)
    arg_parser.add_argument('--rhat_upper_bound', type=float, default=1.1)
    arg_parser.add_argument('--result_dir', type=str, required=True)

    return arg_parser.parse_args()

if __name__ == '__main__':
    main()
