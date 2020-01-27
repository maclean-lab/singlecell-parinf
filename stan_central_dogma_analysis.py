#!/usr/bin/env python
import sys
import os.path
import pickle
import numpy as np
from stan_helpers import StanSampleAnalyzer

def main():
    result_dir = "../../result/stan-central-dogma/local-2"
    y0 = np.zeros(2)
    with open("timePointsCentralDogma.p", "rb") as f:
        ts = pickle.load(f, encoding="bytes")
    with open("dataMatCentralDogma.p", "rb") as f:
        y_ref = pickle.load(f, encoding="bytes")
    y_ref = y_ref.squeeze()

    num_chains = 4
    warmup = 1000
    analyzer = StanSampleAnalyzer(result_dir, num_chains, warmup,
                                  central_dogma_ode, ts, 1, y0, y_ref=y_ref)
    analyzer.simulate_chains()
    analyzer.plot_parameters()

def central_dogma_ode(t, y, theta):
    dydt = np.zeros(2)

    dydt[0] = theta[0] - theta[1] * y[0]
    dydt[1] = theta[2] * y[0] - theta[3] * y[1]

    return dydt

if __name__ == "__main__":
    main()
