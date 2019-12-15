#!/usr/bin/env python
import os.path
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from stan_helpers import StanSampleAnalyzer, moving_average, calcium_ode

def main():
    result_dir = "../../result/stan-calcium-model-cell-3369-1"
    cell_id = 3369
    t0, t_end = 220, 1000
    ts = np.linspace(t0, t_end, t_end - t0 + 1)
    y_ref = np.loadtxt("canorm_tracjectories.csv", delimiter=",")
    y0 = np.array([0, 0, 0.7, y_ref[cell_id, t0]])
    model_name = "calcium_model"
    num_chains = 2
    warmup = 100
    analyzer = StanSampleAnalyzer(result_dir, model_name, num_chains, warmup,
                                  calcium_ode, ts, 3, y0,
                                  y_ref=y_ref[cell_id, t0:])
    analyzer.simulate_chains()
    analyzer.plot_parameters()

if __name__ == "__main__":
    main()
