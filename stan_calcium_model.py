#!/usr/bin/env python
import sys, os.path, pickle
import numpy as np
import scipy.signal
from stan_helpers import StanSession

def filter_trajectory(x):
    sos = scipy.signal.butter(5, 1, btype="lowpass", analog=True,
                              output="sos")
    x_filtered = scipy.signal.sosfilt(sos, x)

    return x_filtered

def main():
    # load model
    if len(sys.argv) < 2:
        print("Usage: python stan_calcium_model.py [stan_model] [result_dir]")
        sys.exit(1)

    calcium_model = sys.argv[1]
    result_dir = "." if len(sys.argv) < 3 else sys.argv[2]

    # load data
    y_raw = np.loadtxt("canorm_tracjectories.csv", delimiter=",")
    t0, t_end = 221, 1000
    y0 = np.array([0, 0, 0.7, y_raw[0, t0]])
    y = y_raw[0, t0 + 1:]
    T = y.size
    ts = np.linspace(t0 + 1, t_end, t_end - t0)
    calcium_data = {
        "N": 4,
        "T": T,
        "y0": y0,
        "y": y,
        "t0": t0,
        "ts": ts,
    }
    print("Data loaded.")

    # set parameters
    num_chains = 4
    num_iters = 1000
    warmup = 1000
    thin = 1

    stan_session = StanSession(calcium_model, calcium_data, result_dir,
                               num_chains=num_chains, num_iters=num_iters,
                               warmup=warmup, thin=thin)
    stan_session.run_sampling()

if __name__ == "__main__":
    main()
