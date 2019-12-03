#!/usr/bin/env python
import sys, os.path, pickle
import numpy as np
from stan_helpers import StanSession

def main():
    # load model
    if len(sys.argv) < 2:
        print("Usage: python stan_central_dogma.py [stan_model] [result_dir]")
        sys.exit(1)

    central_dogma_model = sys.argv[1]
    result_dir = "." if len(sys.argv) < 3 else sys.argv[2]

    # load data
    with open("dataMatCentralDogma.p", "rb") as f:
        y = pickle.load(f, encoding="bytes")
    with open("timePointsCentralDogma.p", "rb") as f:
        ts = pickle.load(f, encoding="bytes")
    central_dogma_data = {
        "N": 2,
        "T": ts.size - 1,
        "y0": np.zeros(2),
        "y": y[:, 1:].squeeze(),
        "t0": ts[0],
        "ts": ts[1:],
    }
    print("Data loaded.")

    # set parameters
    num_chains = 4
    num_iters = 5000
    warmup = 2000
    thin = 1

    stan_session = StanSession(central_dogma_model, central_dogma_data,
                               result_dir, num_chains=num_chains,
                               num_iters=num_iters, warmup=warmup, thin=thin)
    stan_session.run_sampling()

if __name__ == "__main__":
    main()
