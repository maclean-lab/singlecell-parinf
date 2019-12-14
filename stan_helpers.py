import sys
import os.path
import pickle
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from tqdm import tqdm
from pystan import StanModel

class StanSession:
    def __init__(self, stan_model, data, result_dir, num_chains=4,
                 num_iters=1000, warmup=1000, thin=1):
        # load Stan model
        stan_model = os.path.basename(stan_model)
        self.model_name, model_ext = os.path.splitext(stan_model)
        if model_ext == ".stan":
            # load model from Stan code
            self.model = StanModel(file=stan_model, model_name=self.model_name)

            compiled_model_file = os.path.join(result_dir,
                                               self.model_name + ".pkl")
            with open(compiled_model_file, "wb") as f:
                pickle.dump(self.model, f)

            print("Compiled stan model saved.")
        elif model_ext == ".pkl":
            # load saved model
            with open(stan_model, "rb") as f:
                self.model = pickle.load(f)

            print("Compiled stan model loaded.")
        else:
            # cannot load given file, exit
            print("Unsupported input file")
            sys.exit(1)

        self.data = data
        self.result_dir = result_dir
        self.num_chains = num_chains
        self.num_iters = num_iters
        self.warmup = warmup
        self.thin = thin

    def run_sampling(self):
        """Run Stan sampling"""
        # run sampling
        self.fit = self.model.sampling(
            data=self.data, chains=self.num_chains, iter=self.num_iters,
            warmup=self.warmup, thin=self.thin,
            sample_file=os.path.join(self.result_dir, self.model_name))
        print("Sampling finished.")
        sys.stdout.flush()

        # save fit object
        fit_file = os.path.join(self.result_dir, self.model_name + "_fit.pkl")
        with open(fit_file, "wb") as f:
            pickle.dump(self.fit, f)
        print("Stan fit object saved.")

        # plot fit result (native pystan implementation)
        plt.clf()
        self.fit.plot()
        plt.savefig(os.path.join(self.result_dir, self.model_name + "_fit.png"))

        # make trace plot of fit result (arviz API)
        plt.clf()
        az.plot_trace(self.fit)
        plt.savefig(os.path.join(self.result_dir,
                                 self.model_name + "_fit_trace.png"))

class StanSampleAnalyzer():
    """analyze sample files from Stan sampling"""
    theta_0_col = 8

    def __init__(self, result_dir, model_name, num_chains, warmup, ode,
                 timesteps, target_var_idx, y0, y_ref=np.empty(0)):
        self.result_dir = result_dir
        self.model_name = model_name
        self.num_chains = num_chains
        self.warmup = warmup
        self.ode = ode
        self.timesteps = timesteps
        self.target_var_idx = target_var_idx
        self.y0 = y0
        self.y_ref = y_ref

    def simulate_chains(self):
        """analyze each sample file"""
        for chain in range(self.num_chains):
            # load sample file
            sample_file = os.path.join(
                self.result_dir, "{}_{}.csv".format(self.model_name, chain))
            samples = pd.read_csv(sample_file, index_col=False, comment="#")
            thetas = samples.iloc[self.warmup:, self.theta_0_col:].to_numpy()
            num_samples = thetas.shape[0]
            y = np.zeros((num_samples, self.timesteps.size))

            # simulate trajectory from each samples
            print("Simulating trajectories from chain {}".format(chain))
            for sample_idx, theta in tqdm(enumerate(thetas), total=num_samples):
                y[sample_idx, :] = self._simulate_trajectory(theta)

            self._plot_trajectories(chain, y)

    def _simulate_trajectory(self, theta):
        """simulate a trajectory with sampled parameters"""
        # initialize ODE solver
        solver = scipy.integrate.ode(self.ode)
        solver.set_integrator("dopri5")
        solver.set_f_params(theta)
        solver.set_initial_value(self.y0, self.timesteps[0])

        # perform numerical integration
        y = np.zeros_like(self.timesteps)
        y[0] = self.y0[self.target_var_idx]
        i = 1
        while solver.successful() and i < self.timesteps.size:
            solver.integrate(self.timesteps[i])
            y[i] = solver.y[self.target_var_idx]

            i += 1

        return y

    def _plot_trajectories(self, chain_idx, y):
        """plot ODE solution (trajectory)"""
        plt.clf()
        plt.plot(self.timesteps, y.T)

        if self.y_ref.size > 0:
            plt.plot(self.timesteps, self.y_ref, "ko", fillstyle="none")

        figure_name = os.path.join(self.result_dir,
                                   "trajectories_{}.png".format(chain_idx))
        plt.savefig(figure_name)
        plt.close()

    def plot_parameters(self):
        """plot trace for parameters"""
        for chain in range(self.num_chains):
            # load sample file
            sample_file = os.path.join(
                self.result_dir, "{}_{}.csv".format(self.model_name, chain))
            samples = pd.read_csv(sample_file, index_col=False, comment="#")
            samples = samples.to_numpy()

            sigma = samples[self.warmup:, self.theta_0_col - 1]
            theta = samples[self.warmup:, self.theta_0_col:]

            # make trace plot
            plt.clf()
            num_params = theta.shape[1] + 1
            plt.figure(figsize=(6, num_params*2))

            # plot trace of sigma
            plt.subplot(num_params, 1, 1)
            plt.plot(sigma)
            plt.title("sigma")

            # plot trace of theta
            for idx in range(1, num_params):
                plt.subplot(num_params, 1, idx + 1)
                plt.plot(theta[:, idx - 1])
                plt.title("theta[{}]".format(idx - 1))

            plt.tight_layout()

            # save trace plot
            figure_name = os.path.join(self.result_dir,
                                       "parameter_trace_{}.png".format(chain))
            plt.savefig(figure_name)
            plt.close()

            # make violin plot
            plt.clf()
            plt.figure(figsize=(num_params, 4))
            plt.violinplot(samples[self.warmup:, self.theta_0_col - 1:])

            # add paramter names to ticks on x-axis
            param_names = ["sigma"] + \
                ["theta[{}]".format(i) for i in range(0, num_params)]
            param_ticks = np.arange(1, num_params + 1)
            plt.xticks(param_ticks, param_names)

            plt.tight_layout()

            # save violin plot
            figure_name = os.path.join(
                self.result_dir, "parameter_violin_plot_{}.png".format(chain))
            plt.savefig(figure_name)
            plt.close()

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

def filter_trajectory(x):
    """apply a low-pass filter for a trajectory"""
    sos = scipy.signal.butter(5, 1, btype="lowpass", analog=True,
                              output="sos")
    x_filtered = scipy.signal.sosfilt(sos, x)

    return x_filtered

def moving_average(x: np.ndarray, window: int = 20):
    """compute moving average of trajectories"""
    x_df = pd.DataFrame(x)
    x_moving_average = x_df.rolling(window=window, axis=1).mean().to_numpy()

    return x_moving_average
