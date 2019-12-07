#!/usr/bin/env python
import os.path
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from stan_helpers import StanSampleAnalyzer

def calcium_ode(t, y, theta):
    dydt = np.zeros(4)

    dydt[0] = theta[0]* theta[1] * np.exp(-theta[2] * t) - theta[3] * y[0]
    dydt[1] = (theta[4] * y[0] * y[0]) \
        / (theta[5] * theta[5] + y[0] * y[0]) - theta[6] * y[1]
    dydt[2] = theta[7] * (y[3] + theta[8]) \
        * (theta[8] / (y[3] * theta[8]) - y[2])
    beta = 1 + theta[9] * theta[10] / np.power(theta[9] + y[3], 2)
    m_inf = y[1] * y[3] / ((theta[11] + y[1]) * (theta[12] + y[3]))
    dydt[3] = 1 / beta * (
        theta[13]
            * (theta[14] * np.power(m_inf, 3) * np.power(y[2], 3) + theta[15])
            * (theta[17] - (1 + theta[13]) * y[3])
        - theta[16] * np.power(y[3], 2)
            / (np.power(theta[18], 2) + np.power(y[3], 2))
    )

    return dydt

def main():
    result_dir = "../../result/stan-calcium-model-hpc-3"
    t0, t_end = 221, 1000
    ts = np.linspace(t0, t_end, t_end - t0 + 1)
    y_ref = np.loadtxt("canorm_tracjectories.csv", delimiter=",")
    y0 = np.array([0, 0, 0.7, y_ref[0, t0]])
    model_name = "calcium_model"
    num_chains = 4
    analyzer = StanSampleAnalyzer(result_dir, model_name, num_chains,
                                  calcium_ode, ts, 3, y0, y_ref=y_ref[0, t0:])
    # analyzer.simulate_chains()
    analyzer.plot_trace()

if __name__ == "__main__":
    main()
