#!/usr/bin/env python
import argparse
import numpy as np
import scipy.integrate
import pandas as pd
import matplotlib.pyplot as plt
from stan_helpers import calcium_ode_equiv

def main():
    # unpack command line arguments
    args = get_args()
    prior_spec_path = args.prior_spec
    y0 = args.y0
    output_path = args.output

    # load prior specifcation
    prior_spec = pd.read_csv(prior_spec_path, delimiter="\t", index_col=0)
    prior_mean = prior_spec["mu"].to_numpy()
    print("theta = ", prior_mean)

    # simulate the model
    t0, t_end = 0, 800
    ts = np.arange(t0 + 1, t_end + 1)
    y0 = np.array(y0)
    print("y0 =", y0)

    simulate_calcium_ode(prior_mean, t0, ts, y0, output_path)

def simulate_calcium_ode(theta, t0, ts, y0, output_path):
    # initialize ODE solver
    solver = scipy.integrate.ode(calcium_ode_equiv)
    solver.set_integrator("vode", method="bdf")
    solver.set_f_params(theta)
    solver.set_initial_value(y0, t0)

    # perform numerical integration
    y = np.zeros((ts.size, y0.size))
    i = 0
    while solver.successful() and i < ts.size:
        solver.integrate(ts[i])
        y[i, :] = solver.y

        i += 1

    # plot solution
    plt.clf()
    plt.plot(ts, y)
    plt.legend(["PLC", "IP3", "h", "Ca"])
    plt.savefig(output_path)
    plt.close()

def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--prior_spec", type=str, required=True)
    arg_parser.add_argument("--y0", nargs=4, type=float, default=[0, 0, 0, 0])
    arg_parser.add_argument("--output", type=str, required=True)

    return arg_parser.parse_args()

if __name__ == "__main__":
    main()
