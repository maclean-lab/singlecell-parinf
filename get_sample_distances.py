#!/usr/bin/env python
import os.path
import json
import argparse

import numpy as np
import pandas as pd

from stan_helpers import StanMultiSessionAnalyzer
from sample_distance import get_kl_nn
import calcium_models

def main():
    """Load samples and get sample distances
    """
    # parse command-line arguments
    args = get_args()
    stan_runs = args.stan_runs
    list_begin = args.list_begin
    list_end = args.list_end

    with open('stan_run_meta.json', 'r') as f:
        stan_run_meta = json.load(f)

    # load cell chain
    print('Loading samples...')

    param_mask = stan_run_meta[stan_runs[0]]['param_mask']
    param_names = [calcium_models.param_names[i + 1]
                   for i, mask in enumerate(param_mask) if mask == '1']
    param_names = ['sigma'] + param_names

    session_list = []
    session_dirs = []
    for run, l_begin, l_end in zip(stan_runs, list_begin, list_end):
        cell_list_path = os.path.join(
            'cell_lists', stan_run_meta[run]['cell_list'])
        run_cell_list = pd.read_csv(cell_list_path, sep='\t')
        cell_list = run_cell_list.iloc[l_begin:l_end + 1, :]
        run_root = os.path.join('result', stan_run_meta[run]['output_dir'][5:])
        session_list.extend([str(c) for c in cell_list['Cell']])
        session_dirs.extend([os.path.join(run_root, 'samples', f'cell-{c:04d}')
                             for c in cell_list['Cell']])

    if len(stan_runs) == 1:
        output_root = stan_run_meta[stan_runs[0]]['output_dir']
    else:
        output_root = stan_run_meta[stan_runs[0]]['output_dir'][:-2] + '-all'

    output_root = os.path.join('result', output_root[5:])
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    output_dir = os.path.join(output_root, 'multi-sample-analysis')
    analyzer = StanMultiSessionAnalyzer(session_list, output_dir, session_dirs,
                                        param_names=param_names)

    if args.log_normalize:
        session_samples = [np.log1p(a.get_samples().iloc[:, 1:].to_numpy())
                           for a in analyzer.session_analyzers]
    else:
        session_samples = [a.get_samples().iloc[:, 1:].to_numpy()
                           for a in analyzer.session_analyzers]

    num_samples = len(session_samples)
    if args.scale:
        sample_max = np.amax([np.amax(s, axis=0) for s in session_samples],
                             axis=0)
        sample_min = np.amin([np.amin(s, axis=0) for s in session_samples],
                             axis=0)

        for i in range(num_samples):
            session_samples[i] = \
                (session_samples[i] - sample_min) / (sample_max - sample_min)

    print('All samples loaded.')

    # compute distances
    sample_dist_dir = os.path.join(output_root, 'sample-dists')
    if args.log_normalize:
        sample_dist_dir += '-log-normalized'
    if args.scale:
        sample_dist_dir += '-scaled'
    if not os.path.exists(sample_dist_dir):
        os.mkdir(sample_dist_dir)

    print('Computing sample distances using different methods...')

    # Yao 2016 version
    if 'yao' in args.methods:
        print('Current method: Yao 2016')
        sample_dists = get_kl_nn(
            session_samples, method='yao', random_seed=args.random_seed)
        np.save(os.path.join(sample_dist_dir, 'kl_yao.npy'), sample_dists)

    # Yao 2016 version where nan's are replace by 1.0
    if 'yao_1' in args.methods:
        print('Current method: Yao 2016 (nan replaced by 1.0)')
        sample_dists = get_kl_nn(
            session_samples, method='yao', nan_sub=1,
            random_seed=args.random_seed)
        np.save(os.path.join(sample_dist_dir, 'kl_yao_1.npy'), sample_dists)

    # Other KL variants
    for k in args.num_neighbors:
        if 'neighbor_any' in args.methods:
            print(f'Current method: k = {k}; consider any neighbor')
            sample_dists = get_kl_nn(
                session_samples, method='neighbor_any', k=k,
                random_seed=args.random_seed)
            np.save(os.path.join(sample_dist_dir, f'kl_{k}.npy'), sample_dists)

        if 'neighbor_fraction' in args.methods:
            print(f'Current method: k = {k}; consider fraction of neighbors')
            sample_dists = get_kl_nn(
                session_samples, method='neighbor_fraction', k=k,
                random_seed=args.random_seed)
            np.save(os.path.join(sample_dist_dir, f'kl_{k}_frac.npy'),
                    sample_dists)

    print('All sample distances computed and saved.')

def get_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    arg_parser = argparse.ArgumentParser(
        description='Compute distances between posterior samples')
    arg_parser.add_argument('--stan_runs', nargs='+', required=True)
    arg_parser.add_argument('--list_begin', nargs='+', required=True)
    arg_parser.add_argument('--list_end', nargs='+', required=True)
    arg_parser.add_argument('--log_normalize', default=False,
                            action='store_true')
    arg_parser.add_argument('--scale', default=False, action='store_true')
    arg_parser.add_argument('--methods', nargs='+', required=True)
    arg_parser.add_argument('--num_neighbors', nargs='+', default=[2])
    arg_parser.add_argument('--random_seed', type=int, default=0)

    return arg_parser.parse_args()

if __name__ == '__main__':
    main()
