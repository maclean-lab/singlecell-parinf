#!/usr/bin/env python
import os.path
import json
import argparse

import numpy as np
import pandas as pd

from stan_helpers import StanMultiSessionAnalyzer, get_jensen_shannon
import calcium_models

def main():
    # parse command-line arguments
    args = get_args()
    stan_run = args.stan_run
    first_cell_order = args.first_cell_order
    last_cell_order = args.last_cell_order

    # load cell chain
    with open('stan_run_meta.json', 'r') as f:
        stan_run_meta = json.load(f)

    cell_list_path = os.path.join(
        'cell_lists', stan_run_meta[stan_run]['cell_list'])
    full_cell_list = pd.read_csv(cell_list_path, sep='\t')
    cell_list = full_cell_list.iloc[first_cell_order:last_cell_order + 1, :]
    output_root = os.path.join(
        'result', stan_run_meta[stan_run]['output_dir'][5:])
    output_dir = \
        f'multi-sample-analysis-{first_cell_order:04d}-{last_cell_order:04d}'
    session_list = [str(c) for c in cell_list['Cell']]
    session_dirs = [os.path.join(output_root, f'cell-{c:04d}')
                    for c in cell_list['Cell']]
    param_mask = stan_run_meta[stan_run]['param_mask']
    param_names = [calcium_models.param_names[i + 1]
                   for i, mask in enumerate(param_mask) if mask == "1"]
    param_names = ['sigma'] + param_names

    analyzer = StanMultiSessionAnalyzer(session_list, output_dir, session_dirs,
                                        param_names=param_names)

    sample_dist_dir = os.path.join('result', 'sample-dists')
    if not os.path.exists(sample_dist_dir):
        os.mkdir(sample_dist_dir)

    session_samples = [a.get_samples().iloc[:, 1:].to_numpy()
                       for a in analyzer.session_analyzers]

    # compute Jensen-Shannon distances
    js_dists = get_jensen_shannon(session_samples)
    np.save(os.path.join(sample_dist_dir, f'{stan_run}_js.npy'), js_dists)

def get_args():
    arg_parser = argparse.ArgumentParser(
        description='Compute Jensen-Shannon distances for adjacent cells in '
        'a cell chain')
    arg_parser.add_argument('--stan_run', type=str, required=True)
    arg_parser.add_argument('--first_cell_order', type=int, required=True)
    arg_parser.add_argument('--last_cell_order', type=int, required=True)

    return arg_parser.parse_args()

if __name__ == '__main__':
    main()