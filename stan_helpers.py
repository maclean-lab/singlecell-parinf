import sys
import os.path
import itertools
import pickle
import re

import math
import numpy as np
import scipy.integrate
import scipy.stats
import scipy.signal
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import arviz as az
import seaborn as sns

from tqdm import tqdm

from pystan import StanModel
from cmdstanpy import CmdStanModel

class StanSession:
    def __init__(self, stan_model_path, output_dir, stan_backend="pystan",
                 data=None, num_chains=4, num_iters=1000, warmup=1000, thin=1,
                 rhat_upper_bound=1.1):
        # load Stan model
        stan_model_filename = os.path.basename(stan_model_path)
        self.model_name, model_ext = os.path.splitext(stan_model_filename)
        self.stan_backend = stan_backend
        self.output_dir = output_dir
        if model_ext == ".stan":
            # load model from Stan code
            if self.stan_backend == "pystan":
                self.model = StanModel(file=stan_model_path,
                                       model_name=self.model_name)

                compiled_model_path = os.path.join(self.output_dir,
                                                   "stan_model.pkl")
                with open(compiled_model_path, "wb") as f:
                    pickle.dump(self.model, f)

                print("Compiled stan model saved")
            elif self.stan_backend == "cmdstanpy":
                ecompiled_model_path = os.path.join(self.output_dir,
                                                    self.model_name)
                self.model = CmdStanModel(model_name=self.model_name,
                                          stan_file=stan_model_path,
                                          exe_file=compiled_model_path)
            else:
                raise RuntimeError(
                    f"Unsupported Stan backend {self.stan_backend}")
        elif model_ext == ".pkl":
            # load saved model
            with open(stan_model_path, "rb") as f:
                self.model = pickle.load(f)

            print("Compiled stan model loaded")
        else:
            # cannot load given file, exit
            print("Unsupported input file")
            sys.exit(1)

        self.data = data
        self.num_chains = num_chains
        self.num_iters = num_iters
        self.warmup = warmup
        self.thin = thin
        self.rhat_upper_bound = rhat_upper_bound

        sys.stdout.flush()

    def run_sampling(self, control={}):
        """Run Stan sampling"""
        if "max_treedepth" not in control:
            control["max_treedepth"] = 10
        if "adapt_delta" not in control:
            control["adapt_delta"] = 0.8

        # run sampling
        if self.stan_backend == "pystan":
            self.fit = self.model.sampling(
                data=self.data, chains=self.num_chains, iter=self.num_iters,
                warmup=self.warmup, thin=self.thin,
                sample_file=os.path.join(self.output_dir, "chain"),
                control=control)
        else:  # self.stan_backend == "cmdstanpy"
            self.fit = self.model.sample(
                data=self.data, chains=self.num_chains, cores=self.num_chains,
                iter_warmup=self.warmup,
                iter_sampling=self.num_iters - self.warmup, save_warmup=True,
                thin=self.thin, max_treedepth=control["max_treedepth"],
                adapt_delta=control["adapt_delta"], output_dir=self.output_dir)
        print("Stan sampling finished")

        # save fit object
        if self.stan_backend == "pystan":
            stan_fit_path = os.path.join(self.output_dir, "stan_fit.pkl")
            with open(stan_fit_path, "wb") as f:
                pickle.dump(self.fit, f)
            print("Stan fit object saved")

        # convert fit object to arviz's inference data
        print("Converting Stan fit object to Arviz inference data")
        if self.stan_backend == "pystan":
            self.inference_data = az.from_pystan(self.fit)
        else:  # self.stan_backend == "cmdstanpy"
            self.inference_data = az.from_cmdstanpy(self.fit)
        inference_data_path = os.path.join(self.output_dir, "arviz_inf_data.nc")
        az.to_netcdf(self.inference_data, inference_data_path)
        print("Arviz inference data saved")

        sys.stdout.flush()

    def gather_fit_result(self, verbose=True):
        """Run analysis after sampling"""
        if verbose:
            print("Gathering result from stan fit object...")
            sys.stdout.flush()

        # get summary of fit
        if self.stan_backend == "pystan":
            summary_path = os.path.join(self.output_dir, "stan_fit_summary.txt")
            with open(summary_path, "w") as sf:
                sf.write(self.fit.stansummary())

        fit_summary = self.fit.summary()
        if self.stan_backend == "pystan":
            self.fit_summary = pd.DataFrame(
                data=fit_summary["summary"],
                index=fit_summary["summary_rownames"],
                columns=fit_summary["summary_colnames"])
        else:  # self.stan_backend == "cmdstanpy"
            self.fit_summary = fit_summary
        fit_summary_path = os.path.join(self.output_dir, "stan_fit_summary.csv")
        self.fit_summary.to_csv(fit_summary_path)
        if verbose:
            print("Stan summary saved")

        # save samples
        if self.stan_backend == "pystan":
            fit_samples = self.fit.to_dataframe()
            fit_samples_path = os.path.join(self.output_dir,
                                            "stan_fit_samples.csv")
            fit_samples.to_csv(fit_samples_path)

        # TODO: rename sample files from cmdstan

            if verbose:
                print("Stan samples saved")

        # make plots using arviz
        # make trace plot
        plt.clf()
        az.plot_trace(self.inference_data)
        trace_figure_path = os.path.join(self.output_dir, "stan_fit_trace.png")
        plt.savefig(trace_figure_path)
        plt.close()
        if verbose:
            print("Trace plot saved")

        # make plot for posterior
        plt.clf()
        az.plot_posterior(self.inference_data)
        posterior_figure_path = os.path.join(self.output_dir,
                                             "stan_fit_posterior.png")
        plt.savefig(posterior_figure_path)
        plt.close()
        if verbose:
            print("Posterior plot saved")

        # make pair plots
        plt.clf()
        az.plot_pair(self.inference_data)
        pair_figure_path = os.path.join(self.output_dir, "stan_fit_pair.png")
        plt.savefig(pair_figure_path)
        plt.close()
        if verbose:
            print("Pair plot saved")

        sys.stdout.flush()

    def get_good_chain_combo(self):
        """Get a combination of chains with good R_hat value of log
        posteriors
        """
        if 0.9 <= self.fit_summary.loc["lp__", "Rhat"] <= self.rhat_upper_bound:
            return list(range(self.num_chains))

        if self.num_chains <= 2:
            return None

        best_combo = None
        best_rhat = np.inf
        best_rhat_dist = np.inf

        # try remove one bad chain
        for chain_combo in itertools.combinations(range(self.num_chains),
                                                  self.num_chains - 1):
            chain_combo = list(chain_combo)
            combo_data = self.inference_data.sel(chain=chain_combo)
            combo_stats_rhat = az.rhat(combo_data.sample_stats, method="split")
            combo_lp_rhat = combo_stats_rhat["lp"].item()
            combo_lp_rhat_dist = np.abs(combo_lp_rhat - 1.0)

            if combo_lp_rhat_dist < best_rhat_dist:
                best_combo = chain_combo
                best_rhat = combo_lp_rhat
                best_rhat_dist = combo_lp_rhat_dist

        if 0.9 <= best_rhat <= self.rhat_upper_bound:
            return best_combo
        else:
            return None

    def run_optimization(self):
        optimized_params = self.model.optimizing(data=self.data)
        print(optimized_params)

    def run_variational_bayes(self):
        sample_path = os.path.join(self.output_dir, "chain_0.csv")
        diagnostic_path = os.path.join(self.output_dir, "vb_diagnostic.txt")
        vb_results = self.model.vb(data=self.data, sample_file=sample_path,
                                   diagnostic_file=diagnostic_path)

        return vb_results

class StanSessionAnalyzer:
    """Analyze samples from a Stan sampling/varitional Bayes session"""
    def __init__(self, output_dir, stan_backend="pystan",
                 stan_operation="sampling", use_fit_export=False, num_chains=4,
                 warmup=1000, param_names=None, verbose=False):
        self.output_dir = output_dir
        self.stan_backend = stan_backend
        self.stan_operation = stan_operation
        self.use_fit_export = use_fit_export
        self.num_chains = num_chains
        self.warmup = warmup
        self.param_names = param_names

        # load sample files
        if verbose:
            print("Loading stan sample files...")
            sys.stdout.flush()

        self.samples = []
        if self.use_fit_export:
            # get number of chains and number of warmup iterations from
            # stan_fit_summary.txt
            summary_path = os.path.join(self.output_dir, "stan_fit_summary.txt")
            with open(summary_path, "r") as summary_file:
                lines = summary_file.readlines()

            sampling_params = re.findall(r"\d+", lines[1])
            self.num_chains = int(sampling_params[0])
            self.warmup = int(sampling_params[2])

            # get raw samples
            sample_path = os.path.join(self.output_dir, "stan_fit_samples.csv")
            self.raw_samples = pd.read_csv(sample_path, index_col=0)

            # extract sampled parameters
            for chain_idx in range(self.num_chains):
                # TODO: consider drop columns by name, rather than hard code
                # indexing
                samples = self.raw_samples.loc[
                    (self.raw_samples["chain"] == chain_idx)
                     & (self.raw_samples["warmup"] == 0), :
                ].iloc[:, 3:-7]
                samples.set_index(pd.RangeIndex(samples.shape[0]), inplace=True)
                self.samples.append(samples)
        else:
            # use sample files generatd by stan's sampling function
            self.raw_samples = []

            for chain_idx in range(self.num_chains):
                # get raw samples
                sample_path = os.path.join(
                    self.output_dir, "chain_{}.csv".format(chain_idx))
                raw_samples = pd.read_csv(sample_path, index_col=False,
                                          comment="#")
                raw_samples.set_index(pd.RangeIndex(raw_samples.shape[0]),
                                      inplace=True)
                self.raw_samples.append(raw_samples)

                # extract sampled parameters
                if self.stan_operation == "sampling":
                    first_col_idx = 7
                else:
                    first_col_idx = 3
                self.samples.append(
                    raw_samples.iloc[self.warmup:, first_col_idx:])

        self.num_samples = self.samples[0].shape[0]
        self.num_params = self.samples[0].shape[1]

        # set parameters names
        if not param_names or len(param_names) != self.num_params:
            self.param_names = ["sigma"] \
                + ["theta[{}]".format(i) for i in range(self.num_params - 1)]

        for samples in self.samples:
            samples.columns = self.param_names

    def simulate_chains(self, ode, t0, ts, y0, y_ref=None, show_progress=False,
                        var_names=None, integrator="dopri5",
                        **integrator_params):
        """Simulate trajectories for all chains"""
        num_vars = y0.size
        if y_ref is None:
            y_ref = [None] * num_vars
        if var_names is None:
            var_names = [None] * num_vars

        for chain_idx in range(self.num_chains):
            # get thetas
            num_samples = self.samples[chain_idx].shape[0]
            thetas = self.samples[chain_idx].iloc[:num_samples, 1:].to_numpy()
            y = np.zeros((num_samples, ts.size, num_vars))

            # simulate trajectory from each samples
            print("Simulating trajectories from chain {}...".format(chain_idx))
            for sample_idx, theta in tqdm(enumerate(thetas), total=num_samples,
                                          disable=not show_progress):
                y[sample_idx, :, :] = self._simulate_trajectory(
                    ode, theta, t0, ts, y0, integrator=integrator,
                    **integrator_params)

            self._plot_trajectories(chain_idx, ts, y, y_ref, var_names)

            sys.stdout.flush()

    def _simulate_trajectory(self, ode, theta, t0, ts, y0,
                             integrator="dopri5", **integrator_params):
        """Simulate a trajectory with sampled parameters"""
        # initialize ODE solver
        solver = scipy.integrate.ode(ode)
        solver.set_integrator(integrator, **integrator_params)
        solver.set_f_params(theta)
        solver.set_initial_value(y0, t0)

        # perform numerical integration
        y = np.zeros((ts.size, y0.size))
        i = 0
        while solver.successful() and i < ts.size:
            solver.integrate(ts[i])
            y[i, :] = solver.y

            i += 1

        return y

    def _plot_trajectories(self, chain_idx, ts, y, y_ref, var_names):
        """Plot ODE solution (trajectory)"""
        num_vars = len(y_ref)

        plt.clf()
        plt.figure(figsize=(num_vars * 4, 4))

        for var_idx in range(num_vars):
            plt.subplot(1, num_vars, var_idx + 1)
            plt.plot(ts, y[:, :, var_idx].T)

            if y_ref[var_idx] is not None:
                plt.plot(ts, y_ref[var_idx], "ko", fillstyle="none")

            if var_names[var_idx]:
                plt.title(var_names[var_idx])

        plt.tight_layout()
        figure_name = os.path.join(
            self.output_dir, "chain_{}_trajectories.png".format(chain_idx))
        plt.savefig(figure_name)
        plt.close()

    def plot_parameters(self):
        """Make plots for sampled parameters"""
        for chain_idx in range(self.num_chains):
            print("Making trace plot for chain {}...".format(chain_idx))
            self._make_trace_plot(chain_idx)

            print("Making violin plot for chain {}...".format(chain_idx))
            self._make_violin_plot(chain_idx)

            print("Making pairs plot for chain {}...".format(chain_idx))
            self._make_pair_plot(chain_idx)

            sys.stdout.flush()
            sys.stderr.flush()

    def _make_trace_plot(self, chain_idx):
        """Make trace plots for parameters"""
        samples = self.samples[chain_idx].to_numpy()

        plt.clf()
        plt.figure(figsize=(6, self.num_params * 2))

        # plot trace of each parameter
        for idx in range(self.num_params):
            plt.subplot(self.num_params, 1, idx + 1)
            plt.plot(samples[:, idx])
            plt.title(self.param_names[idx])

        plt.tight_layout()

        # save trace plot
        figure_name = os.path.join(
            self.output_dir, "chain_{}_parameter_trace.png".format(chain_idx))
        plt.savefig(figure_name)
        plt.close()

    def _make_violin_plot(self, chain_idx, use_log_scale=True):
        """Make violin plot for parameters"""
        plt.clf()
        plt.figure(figsize=(self.num_params, 4))
        if use_log_scale:
            plt.yscale("log")
        plt.violinplot(self.samples[chain_idx].to_numpy())

        # add paramter names to ticks on x-axis
        param_ticks = np.arange(1, self.num_params + 1)
        plt.xticks(param_ticks, self.param_names)

        plt.tight_layout()

        # save violin plot
        figure_name = os.path.join(
            self.output_dir,
            "chain_{}_parameter_violin_plot.png".format(chain_idx)
        )
        plt.savefig(figure_name)
        plt.close()

    def _make_pair_plot(self, chain_idx):
        """Make pair plot for parameters"""
        plt.clf()
        sns.pairplot(self.samples[chain_idx], diag_kind="kde",
                     plot_kws=dict(alpha=0.4, s=30, color="#191970",
                                   edgecolor="#ffffff", linewidth=0.2),
                     diag_kws=dict(color="#191970", shade=True))
        figure_name = os.path.join(
            self.output_dir,
            "chain_{}_parameter_pair_plot.png".format(chain_idx)
        )
        plt.savefig(figure_name)
        plt.close()

    def get_r_squared(self):
        """Compute R^2 for all pairs of sampled parameters in each chain"""
        for chain_idx in range(self.num_chains):
            r_squared = np.ones((self.num_params, self.num_params))
            for i, j in itertools.combinations(range(self.num_params), 2):
                _, _, r_value, _, _ = scipy.stats.linregress(
                    self.samples[chain_idx].iloc[:, i],
                    self.samples[chain_idx].iloc[:, j]
                )
                r_squared[i, j] = r_squared[j, i] = r_value ** 2

            r_squared_df = pd.DataFrame(r_squared, index=self.param_names,
                                        columns=self.param_names)
            r_squared_df.to_csv(
                os.path.join(self.output_dir,
                             "chain_{}_r_squared.csv".format(chain_idx)),
                float_format="%.8f"
            )

class StanMultiSessionAnalyzer:
    def __init__(self, session_list, result_root, session_output_dirs,
                 use_fit_export=False, num_chains=4, warmup=1000,
                 param_names=None):
        self.session_list = session_list
        self.result_root = result_root
        self.session_output_dirs = session_output_dirs
        self.use_fit_export = use_fit_export
        self.num_chains = num_chains
        self.warmup = warmup
        self.param_names = param_names

        self.num_sessions = len(self.session_list)

        # initialize sample anaylzers for all cells
        self.sample_analyzers = [None] * self.num_sessions
        for i, output_dir in enumerate(self.session_output_dirs):
            self.sample_analyzers[i] = StanSessionAnalyzer(
                os.path.join(self.result_root, output_dir),
                use_fit_export=self.use_fit_export, num_chains=self.num_chains,
                warmup=self.warmup, param_names=self.param_names)

        # make a directory for result
        self.analyzer_result_dir = os.path.join(self.result_root,
                                                "multi_sample_analysis")
        if not os.path.exists(self.analyzer_result_dir):
            os.mkdir(self.analyzer_result_dir)

    def plot_parameter_violin(self, show_progress=False):
        """make violin plots of all parameters"""
        # gather all samples
        all_samples = [np.vstack(analyzer.samples)
                       for analyzer in self.sample_analyzers]
        all_samples = np.array(all_samples)
        all_samples = all_samples.T
        output_path = os.path.join(self.analyzer_result_dir, "param_violin.pdf")

        pdf_multi_plot(plt.violinplot, all_samples, output_path, num_rows=4,
                       num_cols=1, titles=self.param_names,
                       xticks=self.session_list, xtick_rotation=90,
                       show_progress=show_progress)

# utility functions
def calcium_ode_vanilla(t, y, theta):
    """Original calcium model from Yao 2016"""
    dydt = np.zeros(4)

    dydt[0] = theta[0] * theta[1] * np.exp(-theta[2] * t) - theta[3] * y[0]
    dydt[1] = (theta[4] * y[0] * y[0]) \
        / (theta[5] * theta[5] + y[0] * y[0]) - theta[6] * y[1]
    dydt[2] = theta[7] * (y[3] + theta[8]) \
        * (theta[8] / (y[3] * theta[8]) - y[2])
    beta_inv = 1 + theta[9] * theta[10] / np.power(theta[9] + y[3], 2)
    m_inf = y[1] * y[3] / ((theta[11] + y[1]) * (theta[12] + y[3]))
    dydt[3] = 1 / beta_inv * (
        theta[13]
            * (theta[14] * np.power(m_inf, 3) * np.power(y[2], 3) + theta[15])
            * (theta[17] - (1 + theta[13]) * y[3])
        - theta[16] * y[3] * y[3] / (np.power(theta[18], 2) + y[3] * y[3])
    )

    return dydt

def calcium_ode_equiv_1(t, y, theta):
    """Calcium model with equivalent ODEs"""
    dydt = np.zeros(4)

    dydt[0] = theta[0] * theta[1] * np.exp(-theta[2] * t) - theta[3] * y[0]
    dydt[1] = (theta[4] * y[0] * y[0]) \
        / (theta[5] + y[0] * y[0]) - theta[6] * y[1]
    dydt[2] = theta[7] * (theta[8] - (y[3] + theta[8]) * y[2])
    beta = np.power(theta[9] + y[3], 2) \
        / (np.power(theta[9] + y[3], 2) + theta[9] * theta[10])
    m_inf = y[1] * y[3] / ((theta[11] + y[1]) * (theta[12] + y[3]))
    dydt[3] = beta * (
        theta[13]
            * (theta[14] * np.power(m_inf, 3) * np.power(y[2], 3) + theta[15])
            * (theta[17] - (1 + theta[13]) * y[3])
        - theta[16] * y[3] * y[3] / (theta[18] + y[3] * y[3])
    )

    return dydt

def calcium_ode_equiv_2(t, y, theta):
    """Calcium model with equivalent ODEs"""
    dydt = np.zeros(4)

    dydt[0] = theta[0] * np.exp(-theta[1] * t) - theta[2] * y[0]
    dydt[1] = (theta[3] * y[0] * y[0]) \
        / (theta[4] + y[0] * y[0]) - theta[5] * y[1]
    dydt[2] = theta[6] * (theta[7] - (y[3] + theta[7]) * y[2])
    beta = np.power(theta[8] + y[3], 2) \
        / (np.power(theta[8] + y[3], 2) + theta[8] * theta[9])
    m_inf = y[1] * y[3] / ((theta[10] + y[1]) * (theta[11] + y[3]))
    dydt[3] = beta * (
        theta[12]
            * (theta[13] * np.power(m_inf, 3) * np.power(y[2], 3) + theta[14])
            * (theta[16] - (1 + theta[12]) * y[3])
        - theta[15] * y[3] * y[3] / (theta[17] + y[3] * y[3])
    )

    return dydt

def calcium_ode_const_1(t, y, theta):
    """Calcium model with d_1 set to constants"""
    dydt = np.zeros(4)

    dydt[0] = theta[0] * theta[1] * np.exp(-theta[2] * t) - theta[3] * y[0]
    dydt[1] = (theta[4] * y[0] * y[0]) \
        / (theta[5] + y[0] * y[0]) - theta[6] * y[1]
    dydt[2] = theta[7] * (theta[8] - (y[3] + theta[8]) * y[2])
    beta = np.power(theta[9] + y[3], 2) \
        / (np.power(theta[9] + y[3], 2) + theta[9] * theta[10])
    m_inf = y[3] / (theta[11] + y[3])
    dydt[3] = beta * (
        theta[12]
            * (theta[13] * np.power(m_inf, 3) * np.power(y[2], 2) + theta[14])
            * (theta[16] - (1 + theta[12]) * y[3])
        - theta[15] * y[3] * y[3] / (theta[17] + y[3] * y[3])
    )

    return dydt

def calcium_ode_const_2(t, y, theta):
    """Calcium model with d_1 and d_5 set to constants"""
    dydt = np.zeros(4)

    dydt[0] = theta[0] * theta[1] * np.exp(-theta[2] * t) - theta[3] * y[0]
    dydt[1] = (theta[4] * y[0] * y[0]) \
        / (theta[5] + y[0] * y[0]) - theta[6] * y[1]
    dydt[2] = theta[7] * (theta[8] - (y[3] + theta[8]) * y[2])
    beta = np.power(theta[9] + y[3], 2) \
        / (np.power(theta[9] + y[3], 2) + theta[9] * theta[10])
    m_inf = y[1] * y[3] / ((0.13 + y[1]) * (0.0823 + y[3]))
    dydt[3] = beta * (
        theta[11]
            * (theta[12] * np.power(m_inf, 3) * np.power(y[2], 2) + theta[13])
            * (theta[15] - (1 + theta[11]) * y[3])
        - theta[14] * y[3] * y[3] / (theta[16] + y[3] * y[3])
    )

    return dydt

def calcium_ode_const_3(t, y, theta):
    """Calcium model with equivalent ODEs and d_1 set to constant"""
    dydt = np.zeros(4)

    dydt[0] = theta[0] * np.exp(-theta[1] * t) - theta[2] * y[0]
    dydt[1] = (theta[3] * y[0] * y[0]) \
        / (theta[4] + y[0] * y[0]) - theta[5] * y[1]
    dydt[2] = theta[6] * (theta[7] - (y[3] + theta[7]) * y[2])
    beta = np.power(theta[8] + y[3], 2) \
        / (np.power(theta[8] + y[3], 2) + theta[8] * theta[9])
    m_inf = y[1] * y[3] / ((0.13 + y[1]) * (theta[11] + y[3]))
    dydt[3] = beta * (
        theta[11]
            * (theta[12] * np.power(m_inf, 3) * np.power(y[2], 3) + theta[13])
            * (theta[15] - (1 + theta[11]) * y[3])
        - theta[14] * y[3] * y[3] / (theta[16] + y[3] * y[3])
    )

    return dydt

def calcium_ode_reduced(t, y, theta):
    """Calcium model with h and Ca2+ only"""
    dydt = np.zeros(2)

    dydt[0] = theta[0] * (theta[1] - (y[1] + theta[1]) * y[0])
    beta = np.power(theta[2] + y[1], 2) \
        / (np.power(theta[2] + y[1], 2) + theta[2] * theta[3])
    m_inf = theta[4] * y[1] / (theta[5] + y[1])
    dydt[1] = beta * (
        theta[6]
            * (theta[7] * np.power(m_inf, 3) * np.power(y[0], 3) + theta[8])
            * (theta[10] - (1 + theta[6]) * y[1])
        - theta[9] * y[1] * y[1] / (theta[11] + y[1] * y[1])
    )

    return dydt

def load_trajectories(t0, filter_type=None, moving_average_window=20,
                      downsample_offset=-1, downsample_factor=10,
                      verbose=False):
    """Preprocess raw trajectories with filter and downsampling"""
    if verbose:
        print("Loading calcium trajectories...")
    y = np.loadtxt("canorm_tracjectories.csv", delimiter=",")

    # filter the trajectories
    if filter_type == "moving_average":
        y = moving_average(y, window=moving_average_window, verbose=verbose)
    elif filter_type is not None and verbose:
        print(f"Unsupported filter {filter_type}. The trajectories will not "
              + "be filtered.")

    # downsample the trajectories
    t_end = y.shape[1] - 1
    if downsample_offset >= 0 and downsample_factor > 1:
        ts = np.concatenate((
            np.arange(t0 + 1, downsample_offset),
            np.arange(downsample_offset, t_end + 1, downsample_factor)
        ))
    else:
        ts = np.arange(t0 + 1, t_end + 1)

    y0 = y[:, t0]
    y = y[:, ts]
    ts -= t0

    return y, y0, ts

def moving_average(x: np.ndarray, window: int = 20, verbose=True):
    """Compute moving average of trajectories"""
    if verbose:
        print("Performing moving average with window size of "
              + "{}...".format(window))
        sys.stdout.flush()

    # make x 2D if it is 1D
    num_dims = len(x.shape)
    if num_dims == 1:
        x = x[np.newaxis, :]

    x_df = pd.DataFrame(x)
    x_moving_average = x_df.rolling(window=window, axis=1).mean().to_numpy()

    # restore dimension if the given sequence was 1D
    if num_dims == 1:
        x_moving_average = np.squeeze(x_moving_average)

    return x_moving_average

def load_stan_sample_files(sample_dir, num_chains, include_warmup_iters=False):
    """Load samples from sample files generated by Stan's sampler"""
    stan_samples = []

    for chain_idx in range(num_chains):
        sample_path = os.path.join(sample_dir, f"chain_{chain_idx}.csv")

        # get number of warm-up iterations
        num_warmup_iters = 0
        if not include_warmup_iters:
            with open(sample_path, "r") as sf:
                for line in sf:
                    if "warmup=" in line:
                        num_warmup_iters = int(line.strip().split("=")[-1])
                        break

        # get samples
        chain_samples = pd.read_csv(sample_path, index_col=False, comment="#")
        stan_samples.append(chain_samples.iloc[num_warmup_iters:, 7:])

    return stan_samples

def load_stan_fit_samples(sample_file, include_warmup_iters=False):
    """Load samples output by StanFit4Model"""
    stan_samples = []
    raw_samples = pd.read_csv(sample_file, index_col=0)
    chains = np.unique(raw_samples["chain"])

    # add samples for each chain
    for chain_idx in chains:
        samples = raw_samples.loc[raw_samples["chain"] == chain_idx, :]
        # remove rows for warmup iterations
        if not include_warmup_iters:
            samples = samples.loc[samples["warmup"] == 0, :]
        # keep columns for parameters only
        samples = samples.iloc[3:-7]
        # reset row indices
        samples.set_index(pd.RangeIndex(samples.shape[0]), inplace=True)
        stan_samples.append(samples)

    return stan_samples

def load_arviz_inference_data(data_file):
    """Load samples from Arviz's InferenceData"""
    stan_samples = []

    # get samples from saved InferenceData
    inf_data = az.from_netcdf(data_file)
    num_chains = inf_data.sample_stats.dims["chain"]
    sigma_array = inf_data.posterior["sigma"].values
    theta_array = inf_data.posterior["theta"].values

    # gather samples for each chain
    for chain_idx in range(num_chains):
        samples = np.concatenate((sigma_array[chain_idx, :, np.newaxis],
                                  theta_array[chain_idx, :, :]), axis=1)
        stan_samples.append(pd.DataFrame(samples))

    return stan_samples

def get_prior_from_samples(prior_dir, prior_chains,
                           sample_source="arviz_inf_data", verbose=True):
    """Get prior distribution from sampled parameters"""
    if verbose:
        chain_str = ", ".join(map(str, prior_chains))
        print(f"Getting prior distribution from chain(s) {chain_str}...")
        sys.stdout.flush()

    # load sampled parameters
    if sample_source == "sample_files":
        stan_samples = load_stan_sample_files(prior_dir, max(prior_chains) + 1)
    elif sample_source == "fit_export":
        fit_sample_file = os.path.join(prior_dir, "stan_fit_samples.csv")
        stan_samples = load_stan_fit_samples(fit_sample_file)
    else:  # sample_source == Arviz InferenceData
        inf_data_file = os.path.join(prior_dir, "arviz_inf_data.nc")
        stan_samples = load_arviz_inference_data(inf_data_file)

    # retrieve samples from specified chains
    prior_thetas = [samples.iloc[:, 1:]
                    for chain_idx, samples in enumerate(stan_samples)
                    if chain_idx in prior_chains]
    prior_thetas_combined = pd.concat(prior_thetas)

    # compute prior mean and standard deviation
    prior_mean = prior_thetas_combined.mean().to_numpy()
    prior_std = prior_thetas_combined.std().to_numpy()

    return prior_mean, prior_std

def pdf_multi_plot(plot_func, plot_data, output_path, *args, num_rows=4,
                   num_cols=2, titles=None, xticks=None, xtick_rotation=0,
                   show_progress=False):
    """make multiple plots in a PDF"""
    num_subplots_per_page = num_rows * num_cols
    num_plots = len(plot_data)
    num_pages = math.ceil(num_plots / num_subplots_per_page)
    if xticks is not None:
        xtick_pos = np.arange(1, len(xticks) + 1)

    with PdfPages(output_path) as pdf:
        # generate each page
        for page in tqdm(range(num_pages), disable=not show_progress):
            # set page size as US letter
            plt.figure(figsize=(8.5, 11))

            # set number of subplots in current page
            if page == num_pages - 1:
                num_subplots = (num_plots - 1) % num_subplots_per_page + 1
            else:
                num_subplots = num_subplots_per_page

            # make each subplot
            for plot_idx in range(num_subplots):
                data_idx = page * num_subplots_per_page + plot_idx
                plt.subplot(num_rows, num_cols, plot_idx + 1)
                plot_func(plot_data[data_idx], *args)

                if xticks is not None:
                    plt.xticks(xtick_pos, xticks, rotation=xtick_rotation)

                if titles is not None:
                    plt.title(titles[data_idx])

            plt.tight_layout()
            pdf.savefig()
            plt.close()
