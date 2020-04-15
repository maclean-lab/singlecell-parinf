#!/usr/bin/env python
import os.path
from argparse import ArgumentParser
import numpy as np
from scipy.stats import ks_2samp
import pandas as pd
import matplotlib.pyplot as plt
from stan_helpers import StanSessionAnalyzer, moving_average, calcium_ode, \
    pdf_multi_plot

param_names = ["sigma", "KonATP", "L", "Katp", "KoffPLC", "Vplc", "Kip3",
                "KoffIP3", "a", "dinh", "Ke", "Be", "d1", "d5", "epr", "eta1",
                "eta2", "eta3", "c0", "k3"]

def main():
    args = get_args()
    cell_id = args.cell_id
    result_dir = args.result_dir
    session_list_path = args.session_list

    y = np.loadtxt("canorm_tracjectories.csv", delimiter=",")
    y = y[cell_id, :]
    t0 = 200
    t_end = y.size - 1
    ts = np.linspace(t0 + 1, t_end, t_end - t0)
    # apply moving average filter
    y = moving_average(y, window=20)
    y = np.squeeze(y)
    # downsample trajectories
    t_downsample = 300
    y = np.concatenate((y[0:t_downsample], y[t_downsample::10]))
    ts = np.concatenate((ts[0:t_downsample-t0], ts[t_downsample-t0::10]))
    y0 = np.array([0, 0, 0.7, y[t0]])
    y_ref = y[t0 + 1:]

    # create analyzers for all sessions
    session_list = pd.read_csv(session_list_path, delimiter="\t")
    session_list = session_list.astype(str)
    num_sessions = session_list.shape[0]
    session_analyzers = []
    session_chains = []
    for idx in range(num_sessions):
        # create a session analyzer for current session
        analyzer = StanSessionAnalyzer(
            session_list.loc[idx, "dir"], calcium_ode, 3, y0, t0, ts,
            use_summary=True, param_names=param_names, y_ref=y_ref)
        session_analyzers.append(analyzer)

        # add chains of current session
        chains = session_list.loc[idx, "chains"]
        session_chains.append([int(c) for c in chains.split(",")])

    compare_params(session_analyzers, session_list["id"],
                   session_chains, result_dir)

def compare_params(analyzers, session_ids, chain_list, result_dir):
    """make violin plots for parameters"""
    all_samples = []
    num_sessions = len(session_ids)
    xticks = []

    for param in param_names:
        param_samples = []

        # go over each chain in a session
        for idx in range(num_sessions):
            for chain in chain_list[idx]:
                test_sample = analyzers[idx].samples[chain][param].to_numpy()
                param_samples.append(test_sample)

        all_samples.append(param_samples)

    for idx in range(num_sessions):
        for chain in chain_list[idx]:
            xticks.append(f"{session_ids[idx]}:{chain}")

    # make violin plots for all parameters
    figure_path = os.path.join(result_dir, "param_violin.pdf")
    pdf_multi_plot(plt.violinplot, all_samples, figure_path, num_rows=4,
                   num_cols=1, titles=param_names, xticks=xticks)

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
    arg_parser.add_argument("--cell_id", dest="cell_id", type=int,
                            required=True)
    arg_parser.add_argument("--session_list", dest="session_list", type=str,
                            required=True)
    arg_parser.add_argument("--result_dir", dest="result_dir", type=str,
                            required=True)
    arg_parser.add_argument("--filter_type", dest="filter_type",
                            choices=["none", "moving_average"], default="none")
    arg_parser.add_argument("--moving_average_window",
                            dest="moving_average_window", type=int, default=20)

    return arg_parser.parse_args()

if __name__ == "__main__":
    main()
