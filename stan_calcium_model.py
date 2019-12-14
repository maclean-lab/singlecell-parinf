#!/usr/bin/env python
import sys
import os.path
import pickle
import numpy as np
import scipy.signal
import pandas as pd
from stan_helpers import StanSession, moving_average

def main():
    # load model
    if len(sys.argv) < 2:
        print("Usage: python stan_calcium_model.py [stan_model] [result_dir]")
        sys.exit(1)

    calcium_model = sys.argv[1]
    result_dir = "." if len(sys.argv) < 3 else sys.argv[2]

    # prepare data for Stan model
    # load trajectories
    y_raw = np.loadtxt("canorm_tracjectories.csv", delimiter=",")
    t0, t_end = 221, 1000
    y0 = np.array([0, 0, 0.7, y_raw[0, t0]])
    y = y_raw[0, t0 + 1:]
    T = y.size
    ts = np.linspace(t0 + 1, t_end, t_end - t0)

    # get prior distribution from a previous run
    prior_samples = pd.read_csv(
        "../../result/stan-calcium-model-hpc-4/calcium_model_2.csv",
        index_col=False, comment="#")
    prior_warmup = 1000
    prior_theta_0_col = 8
    prior_theta = prior_samples.iloc[prior_warmup:, prior_theta_0_col:]
    prior_mean = prior_theta.mean().to_numpy()
    prior_std = prior_theta.std().to_numpy()

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
    print("Data loaded.")

    # set parameters
    num_chains = 4
    num_iters = 3000
    warmup = 1000
    thin = 1

    stan_session = StanSession(calcium_model, calcium_data, result_dir,
                               num_chains=num_chains, num_iters=num_iters,
                               warmup=warmup, thin=thin)
    stan_session.run_sampling()

if __name__ == "__main__":
    main()
