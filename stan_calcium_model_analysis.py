#!/usr/bin/env python
import os.path
import argparse
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from stan_helpers import StanSampleAnalyzer, calcium_ode

def get_args():
    """parse command line arguments"""
    arg_parser = argparse.ArgumentParser(description="Analyze Stan sample " +
                                         "files.")
    arg_parser.add_argument("--result_dir", dest="result_dir", metavar="DIR",
                            type=str, required=True)
    arg_parser.add_argument("--cell_id", dest="cell_id", metavar="N", type=int,
                            default=0)
    arg_parser.add_argument("--t0", dest="t0", metavar="T", type=int,
                            default=200)
    arg_parser.add_argument("--num_chains", dest="num_chains", type=int,
                            default=4)
    arg_parser.add_argument("--warmup", dest="warmup", type=int, default=1000)
    arg_parser.add_argument("--show_progress", dest="show_progress",
                            default=False, action="store_true")

    return arg_parser.parse_args()

def main():
    # unpack arguments
    args = get_args()
    result_dir = args.result_dir
    cell_id = args.cell_id
    t0 = args.t0
    t_end = 1000
    num_chains = args.num_chains
    warmup = args.warmup
    show_progress = args.show_progress

    # initialize Stan analyzer
    y_ref = np.loadtxt("canorm_tracjectories.csv", delimiter=",")
    y0 = np.array([0, 0, 0.7, y_ref[cell_id, t0]])
    ts = np.linspace(t0, t_end, t_end - t0 + 1)
    analyzer = StanSampleAnalyzer(result_dir, num_chains, warmup, calcium_ode,
                                  ts, 3, y0, y_ref=y_ref[cell_id, t0:],
                                  show_progress=show_progress)

    # run analyses
    analyzer.simulate_chains()
    analyzer.plot_parameters()

if __name__ == "__main__":
    main()
