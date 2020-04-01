#!/usr/bin/env python
import os.path
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
    cell_id = 5106
    ref_dir = f"../../result/stan-calcium-model-100-8/cell-{cell_id}"
    test_dirs = [
        f"../../result/stan-calcium-model-alt-prior-500/cell-{cell_id}",
        f"../../result/stan-calcium-model-alt-prior-1000/cell-{cell_id}"
    ]

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

    result_dir = "../../result/stan-informative-vs-uninformative/" + \
        f"cell-{cell_id}"
    violin_plot_names = ["informative-vs-500.pdf", "informative-vs-1000.pdf"]
    ks_table_names = ["informative-vs-500.csv", "informative-vs-1000.csv"]
    trajectory_dirs = ["alt-trajectories-500", "alt-trajectories-1000"]

    ref_analyzer = StanSessionAnalyzer(ref_dir, calcium_ode, 3, y0, t0, ts,
                                       use_summary=True,
                                       param_names=param_names)

    for i, test_dir in enumerate(test_dirs):
        test_analyzer = StanSessionAnalyzer(test_dir, calcium_ode, 3, y0, t0,
                                            ts, use_summary=True,
                                            param_names=param_names,
                                            y_ref=y_ref)
        compare_sessions(ref_analyzer, test_analyzer,
                         os.path.join(result_dir, violin_plot_names[i]),
                         os.path.join(result_dir, ks_table_names[i]))
        # alt_params = ["d1"]
        # plot_alt_trajectories(ref_analyzer, test_analyzer, alt_params,
        #                       os.path.join(result_dir, trajectory_dirs[i]))

def compare_sessions(ref_analyzer, test_analyzer, plot_name, table_name):
    result_table = pd.DataFrame(columns=["chain", "param", "stat", "p-value"])
    all_samples = []

    for param in param_names:
        param_samples = []

        # get samples from reference session
        ref_sample = pd.concat(
            [sample[param] for sample in ref_analyzer.samples]).to_numpy()
        param_samples.append(ref_sample)

        # go over each chain in test session
        for chain in range(test_analyzer.num_chains):
            test_sample = test_analyzer.samples[chain][param].to_numpy()
            param_samples.append(test_sample)

            # run KS-test
            ks_result = ks_2samp(ref_sample, test_sample)
            row = len(result_table)
            result_table.loc[row] = [chain, param, ks_result[0], ks_result[1]]

        all_samples.append(param_samples)

    # make ref vs test violin plots for all parameters
    xticks = ["Ref"] + list(range(test_analyzer.num_chains))
    pdf_multi_plot(plt.violinplot, all_samples, plot_name, num_rows=4,
                   num_cols=1, titles=param_names, xticks=xticks)

    # write KS-test result to table
    result_table.sort_values("chain", inplace=True, kind="mergesort")
    result_table.to_csv(table_name, index=False)

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

if __name__ == "__main__":
    main()
