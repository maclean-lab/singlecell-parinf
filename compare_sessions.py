#!/usr/bin/env python
import os.path
from argparse import ArgumentParser
import numpy as np
# from scipy.stats import ks_2samp
import pandas as pd
import matplotlib.pyplot as plt
from stan_helpers import StanSessionAnalyzer, moving_average,  pdf_multi_plot

def main():
    args = get_args()
    output_dir = args.output_dir

    session_list_path = args.session_list
    param_mask = args.param_mask

    param_names = ["sigma", "KonATP", "L", "Katp", "KoffPLC", "Vplc", "Kip3",
                   "KoffIP3", "a", "dinh", "Ke", "Be", "d1", "d5", "epr",
                   "eta1", "eta2", "eta3", "c0", "k3"]

    if param_mask:
        param_names = [param_names[i + 1] for i, mask in enumerate(param_mask)
                       if mask == "1"]
        param_names = ["sigma"] + param_names

    # create analyzers for all sessions
    session_list = pd.read_csv(session_list_path, delimiter="\t")
    session_list = session_list.astype(str)
    num_sessions = session_list.shape[0]
    session_analyzers = []
    session_chains = []
    for idx in range(num_sessions):
        # create a session analyzer for current session
        analyzer = StanSessionAnalyzer(session_list.loc[idx, "dir"],
                                       sample_source="arviz_inf_data",
                                       param_names=param_names)
        session_analyzers.append(analyzer)

        # add chains of current session
        chains = session_list.loc[idx, "chains"]
        session_chains.append([int(c) for c in chains.split(",")])

    compare_params(session_analyzers, session_list["id"],
                   session_chains, output_dir, param_names)

def compare_params(analyzers, session_ids, chain_list, output_dir,
                   param_names, output_name="param_violin.pdf"):
    """make violin plots for parameters sampled from different Stan
    sessions
    """
    all_samples = []
    num_sessions = len(session_ids)

    for param in param_names:
        param_samples = []

        # go over each chain in a session
        for idx in range(num_sessions):
            for chain in chain_list[idx]:
                chain_sample = analyzers[idx].samples[chain][param].to_numpy()
                param_samples.append(chain_sample)

        all_samples.append(param_samples)

    # make violin plots for all parameters
    figure_path = os.path.join(output_dir, output_name)
    xticks = [f"{session_ids[idx]}:{chain}" for idx in range(num_sessions)
              for chain in chain_list[idx]]
    pdf_multi_plot(plt.violinplot, all_samples, figure_path, num_rows=4,
                   num_cols=1, titles=param_names, xticks=xticks,
                   xtick_rotation=90)

def plot_alt_trajectories(ref_analyzer, test_analyzer, alt_params, alt_dir):
    # update test analyzer
    result_dir_copy = test_analyzer.result_dir

    test_analyzer.result_dir = alt_dir
    test_samples_copy = [sample.copy() for sample in test_analyzer.samples]
    ref_sample = ref_analyzer.samples[0].copy()

    # truncate reference samples
    num_samples = min(ref_sample.shape[0], test_analyzer.samples[0].shape[0])
    ref_sample = ref_sample.iloc[:num_samples, ]

    # replace alternative parameters for each chain
    for chain in range(test_analyzer.num_chains):
        # truncate test samples
        test_analyzer.samples[chain] = \
            test_analyzer.samples[chain].iloc[:num_samples, ]
        # replace parameters
        test_analyzer.samples[chain][alt_params] = ref_sample[alt_params]

    # simulate all chains in test analyzer
    test_analyzer.simulate_chains(show_progress=True)

    # restore test analyzer
    test_analyzer.result_dir = result_dir_copy
    test_analyzer.samples = test_samples_copy

def get_args():
    """parse command line arguments"""
    arg_parser = ArgumentParser(
        description="Compare results of different Stan sessions.")
    arg_parser.add_argument("--session_list", type=str, required=True)
    arg_parser.add_argument("--output_dir", type=str, required=True)
    arg_parser.add_argument("--param_mask", type=str, default=None)

    return arg_parser.parse_args()

if __name__ == "__main__":
    main()
