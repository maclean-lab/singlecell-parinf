import sys
import os
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, GammaRegressor, \
    HuberRegressor

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
import arviz as az
import seaborn as sns

from anndata import AnnData
import scanpy as sc

from tqdm import tqdm

from pystan import StanModel

class StanSession:
    def __init__(self, stan_model_path, output_dir, data=None, num_chains=4,
                 num_iters=1000, warmup=1000, thin=1, control={},
                 rhat_upper_bound=1.1):
        # load Stan model
        stan_model_filename = os.path.basename(stan_model_path)
        self.model_name, model_ext = os.path.splitext(stan_model_filename)
        self.output_dir = output_dir
        if model_ext == '.stan':
            # load model from Stan code
            self.model = StanModel(file=stan_model_path,
                                    model_name=self.model_name)

            compiled_model_path = os.path.join(self.output_dir,
                                                'stan_model.pkl')
            with open(compiled_model_path, 'wb') as f:
                pickle.dump(self.model, f)

            print('Compiled stan model saved')
        elif model_ext == '.pkl':
            # load saved model
            with open(stan_model_path, 'rb') as f:
                self.model = pickle.load(f)

            print('Compiled stan model loaded')
        else:
            # cannot load given file, exit
            print('Unsupported input file')
            sys.exit(1)

        self.data = data
        self.num_chains = num_chains
        self.num_iters = num_iters
        self.warmup = warmup
        self.thin = thin
        self.control = control
        self.rhat_upper_bound = rhat_upper_bound

        sys.stdout.flush()

    def run_sampling(self):
        """Run Stan sampling"""
        if 'max_treedepth' not in self.control:
            self.control['max_treedepth'] = 10
        if 'adapt_delta' not in self.control:
            self.control['adapt_delta'] = 0.8

        # run sampling
        self.fit = self.model.sampling(
            data=self.data, chains=self.num_chains, iter=self.num_iters,
            warmup=self.warmup, thin=self.thin,
            sample_file=os.path.join(self.output_dir, 'chain'),
            control=self.control)
        print('Stan sampling finished')

        # save fit object
        stan_fit_path = os.path.join(self.output_dir, 'stan_fit.pkl')
        with open(stan_fit_path, 'wb') as f:
            pickle.dump(self.fit, f)
        print('Stan fit object saved')

        # convert fit object to arviz's inference data
        print('Converting Stan fit object to Arviz inference data')
        self.inference_data = az.from_pystan(self.fit)
        inference_data_path = os.path.join(self.output_dir, 'arviz_inf_data.nc')
        az.to_netcdf(self.inference_data, inference_data_path)
        print('Arviz inference data saved')

        sys.stdout.flush()

    def gather_fit_result(self, verbose=True):
        """Run analysis after sampling"""
        if verbose:
            print('Gathering result from stan fit object...')
            sys.stdout.flush()

        # get summary of fit
        summary_path = os.path.join(self.output_dir, 'stan_fit_summary.txt')
        with open(summary_path, 'w') as sf:
            sf.write(self.fit.stansummary())

        fit_summary = self.fit.summary()
        self.fit_summary = pd.DataFrame(
            data=fit_summary['summary'],
            index=fit_summary['summary_rownames'],
            columns=fit_summary['summary_colnames'])
        fit_summary_path = os.path.join(self.output_dir, 'stan_fit_summary.csv')
        self.fit_summary.to_csv(fit_summary_path)
        if verbose:
            print('Stan summary saved')

        # save samples
        fit_samples = self.fit.to_dataframe()
        fit_samples_path = os.path.join(self.output_dir,
                                        'stan_fit_samples.csv')
        fit_samples.to_csv(fit_samples_path)

        if verbose:
            print('Stan samples saved')

        # make plots using arviz
        # make trace plot
        plt.clf()
        az.plot_trace(self.inference_data)
        trace_figure_path = os.path.join(self.output_dir, 'stan_fit_trace.png')
        plt.savefig(trace_figure_path)
        plt.close()
        if verbose:
            print('Trace plot saved')

        # make plot for posterior
        plt.clf()
        az.plot_posterior(self.inference_data)
        posterior_figure_path = os.path.join(self.output_dir,
                                             'stan_fit_posterior.png')
        plt.savefig(posterior_figure_path)
        plt.close()
        if verbose:
            print('Posterior plot saved')

        # make pair plots
        plt.clf()
        az.plot_pair(self.inference_data)
        pair_figure_path = os.path.join(self.output_dir, 'stan_fit_pair.png')
        plt.savefig(pair_figure_path)
        plt.close()
        if verbose:
            print('Pair plot saved')

        sys.stdout.flush()

    def get_mixed_chains(self):
        """Get a combination of chains with good R_hat value of log
        posteriors
        """
        if 0.9 <= self.fit_summary.loc['lp__', 'Rhat'] <= self.rhat_upper_bound:
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
            combo_stats_rhat = az.rhat(combo_data.sample_stats[['lp']],
                                       method='split')
            combo_lp_rhat = combo_stats_rhat['lp'].item()
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

        return optimized_params

    def run_variational_bayes(self):
        sample_path = os.path.join(self.output_dir, 'chain_0.csv')
        diagnostic_path = os.path.join(self.output_dir, 'vb_diagnostic.txt')
        vb_results = self.model.vb(data=self.data, sample_file=sample_path,
                                   diagnostic_file=diagnostic_path)

        return vb_results

class StanSessionAnalyzer:
    """Analyze samples from a Stan sampling/varitional Bayes session"""
    def __init__(self, output_dir, stan_operation='sampling',
                 sample_source='arviz_inf_data', num_chains=4, warmup=1000,
                 param_names=None, verbose=False):
        self.output_dir = output_dir
        self.stan_operation = stan_operation
        self.sample_source = sample_source
        self.num_chains = num_chains
        self.warmup = warmup
        self.param_names = param_names

        # load sample files
        if verbose:
            print('Loading stan sample files...')
            sys.stdout.flush()

        self.samples = []
        self.log_posterior = []
        if self.sample_source == 'sample_files':
            # use sample files generatd by stan's sampling function
            self.raw_samples = []

            for chain_idx in range(self.num_chains):
                # get raw samples
                sample_path = os.path.join(
                    self.output_dir, 'chain_{}.csv'.format(chain_idx))
                raw_samples = pd.read_csv(sample_path, index_col=False,
                                          comment='#')
                raw_samples.set_index(pd.RangeIndex(raw_samples.shape[0]),
                                      inplace=True)
                self.raw_samples.append(raw_samples)

                # extract sampled parameters
                if self.stan_operation == 'sampling':
                    first_col_idx = 7
                else:
                    first_col_idx = 3
                self.samples.append(
                    raw_samples.iloc[self.warmup:, first_col_idx:])

                # get log posterior
                self.log_posterior.append(raw_samples.iloc[self.warmup:, 0])
        elif self.sample_source == 'fit_export':
            # get number of chains and number of warmup iterations from
            # stan_fit_summary.txt
            summary_path = os.path.join(self.output_dir, 'stan_fit_summary.txt')
            with open(summary_path, 'r') as summary_file:
                lines = summary_file.readlines()

            sampling_params = re.findall(r'\d+', lines[1])
            self.num_chains = int(sampling_params[0])
            self.warmup = int(sampling_params[2])

            # get samples
            sample_path = os.path.join(self.output_dir, 'stan_fit_samples.csv')
            self.raw_samples = pd.read_csv(sample_path, index_col=0)
            self.samples = load_stan_fit_samples(sample_path)

            # get log posterior
            for chain_idx in range(self.num_chains):
                self.log_posterior.append(
                    self.raw_samples.loc[
                        self.raw_samples['warmup'] == 0 &
                        self.raw_samples['chain'] == chain_idx, :])
        else:  # sample_source == 'arviz_inf_data':
            sample_path = os.path.join(self.output_dir, 'arviz_inf_data.nc')
            self.raw_samples = az.from_netcdf(sample_path)
            self.num_chains = self.raw_samples.sample_stats.dims['chain']
            self.samples = load_arviz_inference_data(sample_path)
            for chain_idx in range(self.num_chains):
                self.log_posterior.append(
                    self.raw_samples.sample_stats['lp'][chain_idx].values)

        self.num_samples = self.samples[0].shape[0]
        self.num_params = self.samples[0].shape[1]

        # set parameters names
        if not param_names or len(param_names) != self.num_params:
            self.param_names = ['sigma'] \
                + [f'theta[{i}]' for i in range(self.num_params - 1)]

        for samples in self.samples:
            samples.columns = self.param_names

    def get_sampling_time(self, unit='s'):
        '''get runtime for all chains'''
        sampling_time = np.zeros(self.num_chains)

        for chain_idx in range(self.num_chains):
            sample_file_path = os.path.join(self.output_dir,
                                            f'chain_{chain_idx}.csv')
            with open(sample_file_path, 'r') as sf:
                time_text = sf.readlines()[-2]
                sampling_time[chain_idx] = float(time_text.split()[1])
                if unit == 'm':
                   sampling_time[chain_idx] /= 60
                elif unit == 'h':
                   sampling_time[chain_idx] /= 3600

        return sampling_time

    def get_tree_depths(self):
        if self.sample_source == 'arviz_inf_data':
            inf_data = self.raw_samples
        else:
            inf_data_path = os.path.join(self.output_dir, 'arviz_inf_data.nc')
            inf_data = az.from_netcdf(inf_data_path)

        return inf_data.sample_stats['treedepth'].data

    def simulate_chains(self, ode, t0, ts, y0, y_ref=None, show_progress=False,
                        var_names=None, integrator='dopri5',
                        subsample_step_size=None, plot=True, verbose=True,
                        **integrator_params):
        """Simulate trajectories for all chains"""
        num_vars = y0.size
        if y_ref is None:
            y_ref = [None] * num_vars
        if var_names is None:
            var_names = [None] * num_vars

        y_sim = []
        for chain_idx in range(self.num_chains):
            # get thetas
            thetas = self.samples[chain_idx].to_numpy()[:, 1:]
            if subsample_step_size:
                thetas = thetas[::subsample_step_size, :]
            num_samples = thetas.shape[0]
            y = np.zeros((num_samples, ts.size, num_vars))

            # simulate trajectory from each samples
            if verbose:
                print(f'Simulating trajectories from chain {chain_idx}...')

            for sample_idx, theta in tqdm(enumerate(thetas), total=num_samples,
                                          disable=not show_progress):
                y[sample_idx, :, :] = simulate_trajectory(
                    ode, theta, t0, ts, y0, integrator=integrator,
                    **integrator_params)

            y_sim.append(y)

            if plot:
                self._plot_trajectories(chain_idx, ts, y, y_ref, var_names)

            sys.stdout.flush()

        return y_sim

    def _plot_trajectories(self, chain_idx, ts, y, y_ref, var_names):
        """Plot ODE solution (trajectory)"""
        num_vars = len(y_ref)

        plt.clf()
        plt.figure(figsize=(num_vars * 4, 4))

        for var_idx in range(num_vars):
            plt.subplot(1, num_vars, var_idx + 1)
            plt.plot(ts, y[:, :, var_idx].T)

            if y_ref[var_idx] is not None:
                plt.plot(ts, y_ref[var_idx], 'ko', fillstyle='none')

            if var_names[var_idx]:
                plt.title(var_names[var_idx])

        plt.tight_layout()
        figure_name = os.path.join(
            self.output_dir, 'chain_{}_trajectories.png'.format(chain_idx))
        plt.savefig(figure_name)
        plt.close()

    def get_trajectory_distance(self, ode, t0, ts, y0, y_ref, target_var_idx,
                                rhat_upper_bound=4.0, subsample_step_size=50,
                                integrator='dopri5', **integrator_params):
        '''Compute distance between a reference trajectory and trajectory
        simulated from subsamples'''
        mixed_samples = self.get_samples(rhat_upper_bound=rhat_upper_bound)
        thetas = mixed_samples.to_numpy()[:, 1:]
        if subsample_step_size:
            thetas = thetas[::subsample_step_size, :]
        y_diffs = np.empty(thetas.shape[0])
        for i, theta in enumerate(thetas):
            y = simulate_trajectory(ode, theta, t0, ts, y0,
                                    integrator=integrator, **integrator_params)
            y_diffs[i] = np.linalg.norm(
                y[:, target_var_idx] - y_ref[target_var_idx])

        return np.mean(y_diffs)

    def get_sample_mean_trajectory(self, ode, t0, ts, y0, rhat_upper_bound=4.0,
                                    integrator='dopri5', **integrator_params):
        sample_means = self.get_sample_means(rhat_upper_bound=rhat_upper_bound)
        thetas = sample_means.to_numpy()[1:]
        y = simulate_trajectory(ode, thetas, t0, ts, y0, integrator=integrator,
                                **integrator_params)

        return y

    def plot_parameters(self):
        """Make plots for sampled parameters"""
        for chain_idx in range(self.num_chains):
            print(f'Making trace plot for chain {chain_idx}...')
            self._make_trace_plot(chain_idx)

            print(f"Making violin plot for chain {chain_idx}...")
            self._make_violin_plot(chain_idx)

            print(f"Making pairs plot for chain {chain_idx}...")
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
            self.output_dir, f'chain_{chain_idx}_parameter_trace.png')
        plt.savefig(figure_name)
        plt.close()

    def _make_violin_plot(self, chain_idx, use_log_scale=True):
        """Make violin plot for parameters"""
        plt.clf()
        plt.figure(figsize=(self.num_params, 4))
        if use_log_scale:
            plt.yscale('log')
        plt.violinplot(self.samples[chain_idx].to_numpy())

        # add paramter names to ticks on x-axis
        param_ticks = np.arange(1, self.num_params + 1)
        plt.xticks(param_ticks, self.param_names)

        plt.tight_layout()

        # save violin plot
        figure_name = os.path.join(
            self.output_dir, f'chain_{chain_idx}_parameter_violin_plot.png')
        plt.savefig(figure_name)
        plt.close()

    def _make_pair_plot(self, chain_idx):
        """Make pair plot for parameters"""
        plt.clf()
        sns.pairplot(self.samples[chain_idx], diag_kind='kde',
                     plot_kws=dict(alpha=0.4, s=30, color='#191970',
                                   edgecolor='#ffffff', linewidth=0.2),
                     diag_kws=dict(color='#191970', shade=True))
        figure_name = os.path.join(
            self.output_dir, f'chain_{chain_idx}_parameter_pair_plot.png')
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
                             f'chain_{chain_idx}_r_squared.csv'),
                float_format='%.8f'
            )

    def get_mixed_chains(self, rhat_upper_bound=4.0, return_rhat=False):
        """Get a combination of chains with good R_hat value of log
        posteriors
        """
        if self.num_chains <= 2 or \
                not isinstance(self.raw_samples, az.InferenceData):
            return None

        sample_rhat = az.rhat(self.raw_samples.sample_stats[['lp']],
                              method='split')
        lp_rhat = sample_rhat['lp'].item()
        if 0.9 <= lp_rhat <= rhat_upper_bound:
            all_chains = list(range(self.num_chains))
            if return_rhat:
                return all_chains, lp_rhat
            else:
                return all_chains

        best_combo = None
        best_rhat = np.inf
        best_rhat_dist = np.inf

        # try remove one bad chain
        for chain_combo in itertools.combinations(range(self.num_chains),
                                                  self.num_chains - 1):
            chain_combo = list(chain_combo)
            combo_data = self.raw_samples.sel(chain=chain_combo)
            combo_stats_rhat = az.rhat(combo_data.sample_stats[['lp']],
                                       method='split')
            combo_lp_rhat = combo_stats_rhat['lp'].item()
            combo_lp_rhat_dist = np.abs(combo_lp_rhat - 1.0)

            if combo_lp_rhat_dist < best_rhat_dist:
                best_combo = chain_combo
                best_rhat = combo_lp_rhat
                best_rhat_dist = combo_lp_rhat_dist

        if 0.9 <= best_rhat <= rhat_upper_bound:
            if return_rhat:
                return best_combo, best_rhat
            else:
                return best_combo
        else:
            if return_rhat:
                return None, None
            else:
                return None

    def get_samples(self, rhat_upper_bound=4.0, excluded_params=None):
        """Get sampled parameters of mixed chains"""
        mixed_chains = self.get_mixed_chains(rhat_upper_bound=rhat_upper_bound)

        if not mixed_chains:
            return None

        mixed_samples = pd.concat([self.samples[c] for c in mixed_chains])

        if excluded_params is not None:
            mixed_samples.drop(labels=excluded_params, axis=1, inplace=True)

        return mixed_samples

    def get_sample_means(self, rhat_upper_bound=4.0):
        """Get means of all sampled parameters in mixed chains"""
        mixed_chains = self.get_mixed_chains(rhat_upper_bound=rhat_upper_bound)

        if not mixed_chains:
            return None

        mixed_samples = pd.concat([self.samples[c] for c in mixed_chains])
        sample_means = mixed_samples.mean()

        return sample_means

    def get_sample_modes(self, method='kde', bins=100, rhat_upper_bound=4.0):
        """Get means of all sampled parameters in mixed chains"""
        mixed_chains = self.get_mixed_chains(rhat_upper_bound=rhat_upper_bound)

        if not mixed_chains:
            return None

        mixed_samples = pd.concat([self.samples[c] for c in mixed_chains])
        sample_modes = pd.Series(index=self.param_names)
        for param in self.param_names:
            sample_modes[param] = get_mode_continuous_rv(
                mixed_samples[param], method=method, bins=bins)

        return sample_modes

    def get_mean_log_posteriors(self):
        """Get mean of log posterior density"""
        return np.array([np.mean(lp) for lp in self.log_posterior])

    def get_log_posteriors(self, include_warmup=True):
        """Get log posterior"""
        # load sample files
        if self.sample_source == 'sample_files':
            samples = self.raw_samples
        else:
            samples = []
            for chain_idx in range(self.num_chains):
                # get raw samples
                sample_path = os.path.join(
                    self.output_dir, 'chain_{}.csv'.format(chain_idx))
                sample = pd.read_csv(sample_path, index_col=False, comment='#')
                sample.set_index(pd.RangeIndex(sample.shape[0]), inplace=True)
                samples.append(sample)

        # remove warmup iterations if specified
        if not include_warmup:
            for i in range(len(samples)):
                samples[i] = samples[i].loc[self.warmup:, :]

        lps = np.array([s['lp__'].to_numpy() for s in samples])

        return lps

    def plot_log_posteriors(self, include_warmup=True):
        """Plot log posterior, including warmup"""
        lps = self.get_log_posteriors(include_warmup=include_warmup)

        plt.figure(figsize=(6, 4), dpi=300)
        plt.plot(lps.T)
        plt.ylim((0, np.amax(lps)))
        figure_path = os.path.join(self.output_dir, 'log_posterior_trace.png')
        plt.savefig(figure_path)
        plt.close()

class StanMultiSessionAnalyzer:
    def __init__(self, session_list, output_dir, session_output_dirs,
                 sample_source='arviz_inf_data', num_chains=4, warmup=1000,
                 param_names=None, rhat_upper_bound=4.0):
        self.output_dir = output_dir
        self.full_session_output_dirs = session_output_dirs
        self.full_session_list = session_list
        self.sample_source = sample_source
        self.num_chains = num_chains
        self.warmup = warmup
        self.param_names = param_names
        self.rhat_upper_bound = rhat_upper_bound

        # initialize sample anaylzers for all cells
        self.session_list = []
        self.session_out_dirs = []
        self.session_analyzers = []
        for session_name, output_dir in zip(session_list, session_output_dirs):
            analyzer = StanSessionAnalyzer(
                output_dir, sample_source=self.sample_source,
                num_chains=self.num_chains, warmup=self.warmup,
                param_names=self.param_names)

            mixed_chains = analyzer.get_mixed_chains(
                rhat_upper_bound=rhat_upper_bound)

            if mixed_chains is not None:
                self.session_list.append(session_name)
                self.session_out_dirs.append(output_dir)
                self.session_analyzers.append(analyzer)

        self.num_sessions = len(self.session_list)
        self.session_list = np.array(self.session_list)

        # set default parameter names if not given
        if self.param_names is None:
            self.num_params = self.session_analyzers[0].samples[0].shape[1]
            self.param_names = ['sigma'] + \
                [f'theta_{i}' for i in range(self.num_params - 1)]
        else:
            self.num_params = len(param_names)

        # make a directory for result
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def get_sample_means(self):
        '''Get sample means for all sessions'''
        self.sample_means = pd.DataFrame(columns=self.param_names)

        for analyzer in self.session_analyzers:
            session_means = analyzer.get_sample_means(
                rhat_upper_bound=self.rhat_upper_bound)
            self.sample_means = self.sample_means.append(session_means,
                                                         ignore_index=True)

    def filter_sessions(self, z_score_max=3.0, plot=False):
        '''Filter sessions by z-scores of sample means

        Sample means will be calculated and assigned as a field of the analyzer
        if not already
        '''
        if not hasattr(self, 'sample_means'):
            self.get_sample_means()

        sample_mean_z_scores = self.sample_means.apply(scipy.stats.zscore,
                                                       ddof=1)
        # plot z-scores on sample means vs gene expression
        if plot and hasattr(self, 'log_data'):
            gvp_dir = os.path.join(self.output_dir, 'genes-vs-params')
            if not os.path.exists(gvp_dir):
                os.mkdir(gvp_dir)

            for gene_idx, gene in enumerate(self.gene_symbols):
                gvp_scatter_path = os.path.join(gvp_dir,
                                                 f'{gene}_z_scores.pdf')
                self._scatter_multi_plot(
                    self.log_data[gene_idx, self.session_list],
                    gvp_scatter_path, c=sample_mean_z_scores)

        is_z_score_low = \
            (sample_mean_z_scores.abs() < z_score_max).all(axis=1)
        self.session_list = self.session_list[is_z_score_low]
        self.session_analyzers = itertools.compress(self.session_analyzers,
                                                    is_z_score_low)
        self.sample_means = self.sample_means.loc[is_z_score_low, :]
        self.num_sessions = len(self.session_list)

    def get_sample_modes(self, method='kde', bins=100, rhat_upper_bound=4.0):
        '''Get modes of all sessions'''
        self.sample_modes = pd.DataFrame(index=self.session_list,
                                 columns=self.param_names)

        for session, analyzer in zip(self.session_list, self.session_analyzers):
            self.sample_modes.loc[session, :] = analyzer.get_sample_modes(
                method=method, bins=bins, rhat_upper_bound=rhat_upper_bound)

    def get_parameter_correlations(self, sort=True, plot=False, num_pairs=20,
                                   num_rows=4, num_cols=2):
        '''Compute correlation between pairs of parameters using sample means'''
        if not hasattr(self, 'sample_means'):
            print('No operation performed since sample means has not been'
                  + 'calculated')
            return

        num_param_param_pairs = self.num_params * (self.num_params - 1) // 2
        self.param_param_corrs = pd.DataFrame(
            columns=['Param1', 'Param2', 'Correlation', 'p-value'],
            index=range(num_param_param_pairs))
        for i, (p1, p2) in enumerate(
            itertools.combinations(self.param_names, 2)):
            corr, p_value = scipy.stats.pearsonr(self.sample_means[p1],
                                                 self.sample_means[p2])
            self.param_param_corrs.iloc[i, :] = [p1, p2, corr, p_value]

        if sort:
            self.param_param_corrs.sort_values(
                'Correlation', ascending=False, inplace=True,
                ignore_index=True, key=lambda x: np.abs(x))

        if plot:
            figure_path = os.path.join(self.output_dir,
                                       'param_param_scatter.pdf')
            self._plot_param_param_scatter(figure_path, num_pairs, num_rows,
                                           num_cols)

    def _plot_param_param_scatter(self, output_path, num_pairs, num_rows,
                                  num_cols):
        '''Make param-param scatter plot'''
        num_subplots_per_page = num_rows * num_cols
        num_plots = num_pairs
        num_pages = math.ceil(num_plots / num_subplots_per_page)
        session_orders = list(range(self.num_sessions))

        with PdfPages(output_path) as pdf:
            for page in range(num_pages):
                plt.figure(figsize=(8.5, 11))

                if page == num_pages - 1:
                    num_subplots = (num_plots - 1) % num_subplots_per_page + 1
                else:
                    num_subplots = num_subplots_per_page

                for plot_idx in range(num_subplots):
                    plt.subplot(num_rows, num_cols, plot_idx + 1)
                    pair_rank = page * num_subplots_per_page + plot_idx

                    param_1 = self.param_param_corrs.loc[pair_rank, 'Param1']
                    param_2 = self.param_param_corrs.loc[pair_rank, 'Param2']
                    corr = self.param_param_corrs.loc[pair_rank, 'Correlation']
                    plt.scatter(self.sample_means[param_1],
                                self.sample_means[param_2], c=session_orders)
                    plt.title(f'{param_1} vs {param_2}: {corr:.6f}')
                    plt.colorbar()

                plt.tight_layout()
                pdf.savefig()
                plt.close()

    def plot_parameter_violin(self, page_size=(8.5, 11),
                              dpi=300, num_rows=4, num_cols=1, titles=None,
                              xticks=None, y_labels=None):
        """make violin plots of all parameters"""
        # gather all samples
        all_samples = [np.vstack(analyzer.samples)
                       for analyzer in self.session_analyzers]
        all_samples = np.array(all_samples)
        all_samples = all_samples.T

        # set up plots
        num_subplots_per_page = num_rows * num_cols
        num_plots = self.num_params
        num_pages = math.ceil(num_plots / num_subplots_per_page)
        if not titles:
            titles = self.param_names
        violin_pos = [self.full_session_list.index(s) + 1
                      for s in self.session_list]

        output_path = os.path.join(self.output_dir, 'param_violin.pdf')
        with PdfPages(output_path) as pdf:
            for page in range(num_pages):
                plt.figure(figsize=page_size, dpi=dpi)

                if page == num_pages - 1:
                    num_subplots = (num_plots - 1) % num_subplots_per_page + 1
                else:
                    num_subplots = num_subplots_per_page

                for plot_idx in range(num_subplots):
                    plt.subplot(num_rows, num_cols, plot_idx + 1)
                    param_idx = page * num_subplots_per_page + plot_idx
                    param_samples = []
                    for analyzer in self.session_analyzers:
                        session_smaples = [s.values[:, param_idx]
                                            for s in analyzer.samples]
                        param_samples.append(
                            np.concatenate(session_smaples))
                    plt.violinplot(param_samples, positions=violin_pos)
                    if xticks is not None:
                        plt.xticks(ticks=xticks['ticks'] + 1,
                                   labels=xticks['labels'])
                    if y_labels is not None:
                        plt.ylabel(y_labels[param_idx])
                    plt.title(titles[param_idx])

                plt.tight_layout()
                pdf.savefig()
                plt.close()

    def plot_parameter_ribbon(self, page_size=(8.5, 11), dpi=300, num_rows=4,
                              num_cols=1, titles=None):
        '''Make ribbon plot for parameters'''
        # make 'long' form for sampled parameters / concat all samples
        all_samples = pd.DataFrame(columns=['Order'] + self.param_names)
        for i, analyzer in enumerate(self.session_analyzers):
            mixed_chains = analyzer.get_mixed_chains()
            if mixed_chains is not None:
                for chain in mixed_chains:
                    chain_sample = analyzer.samples[chain].copy()
                    # add a cell order column
                    chain_sample.insert(0, 'Order', i)
                    all_samples = all_samples.append(chain_sample,
                                                     ignore_index=True)

        # set up plots
        num_subplots_per_page = num_rows * num_cols
        num_plots = self.num_params
        num_pages = math.ceil(num_plots / num_subplots_per_page)
        if not titles:
            titles = self.param_names

        output_path = os.path.join(self.output_dir, 'param_ribbon.pdf')
        with PdfPages(output_path) as pdf:
            for page in range(num_pages):
                plt.figure(figsize=page_size, dpi=dpi)

                if page == num_pages - 1:
                    num_subplots = (num_plots - 1) % num_subplots_per_page + 1
                else:
                    num_subplots = num_subplots_per_page

                for plot_idx in range(num_subplots):
                    plt.subplot(num_rows, num_cols, plot_idx + 1)
                    param_idx = page * num_subplots_per_page + plot_idx
                    param = self.param_names[param_idx]
                    sns.lineplot(data=all_samples, x='Order', y=param, ci='sd')
                    plt.xlabel('')
                    plt.ylabel('')
                    plt.title(titles[param_idx])

                plt.tight_layout()
                pdf.savefig()
                plt.close()

    def plot_parameter_box(self, page_size=(8.5, 11), dpi=300, num_rows=4,
                           num_cols=1, titles=None, xticks=None,
                           y_labels=None):
        '''Make ribbon plot for parameters'''
        # make 'long' form for sampled parameters / concat all samples
        all_samples = pd.DataFrame(columns=['Order'] + self.param_names)
        for i, analyzer in enumerate(self.session_analyzers):
            mixed_chains = analyzer.get_mixed_chains()
            if mixed_chains is not None:
                for chain in mixed_chains:
                    chain_sample = analyzer.samples[chain].copy()
                    # add a cell order column
                    chain_sample.insert(0, 'Order', i)
                    all_samples = all_samples.append(chain_sample,
                                                     ignore_index=True)

        # set up plots
        num_subplots_per_page = num_rows * num_cols
        num_plots = self.num_params
        num_pages = math.ceil(num_plots / num_subplots_per_page)
        if not titles:
            titles = self.param_names

        output_path = os.path.join(self.output_dir, 'param_box.pdf')
        with PdfPages(output_path) as pdf:
            for page in range(num_pages):
                plt.figure(figsize=page_size, dpi=dpi)

                if page == num_pages - 1:
                    num_subplots = (num_plots - 1) % num_subplots_per_page + 1
                else:
                    num_subplots = num_subplots_per_page

                for plot_idx in range(num_subplots):
                    plt.subplot(num_rows, num_cols, plot_idx + 1)
                    param_idx = page * num_subplots_per_page + plot_idx
                    param = self.param_names[param_idx]
                    sns.boxplot(data=all_samples, x='Order', y=param,
                                color='C0', linewidth=0.7, showfliers=False)
                    if xticks is not None:
                        plt.xticks(**xticks)
                    plt.xlabel('')
                    if y_labels is not None:
                        plt.ylabel(y_labels[param_idx])
                    plt.title(titles[param_idx])

                plt.tight_layout()
                pdf.savefig()
                plt.close()

    def plot_param_pairs_all_sessions(self, param_pairs,
                                      output_path_prefixes=None, dpi=300,
                                      param_names_on_plot=None):
        '''Make scatter plot for select pairs of parameters of all sessions,
        with histogram on the sides'''

        session_plot_dir = os.path.join(self.output_dir, 'param-pairs')
        if not os.path.exists(session_plot_dir):
            os.mkdir(session_plot_dir)

        hist_bins = 10  # number of bins of historgram on the sides

        for i, analyzer in enumerate(tqdm(self.session_analyzers)):
            session_samples = analyzer.get_samples()
            for p1, p2 in param_pairs:
                if output_path_prefixes:
                    output_path = f'{output_path_prefixes[i]}_{p1}_{p2}.pdf'
                else:
                    output_path = os.path.join(
                        session_plot_dir,
                        f'{i:04d}_{self.session_list[i]:04d}_{p1}_{p2}.pdf')

                p1_samples = session_samples.loc[:, p1]
                p2_samples = session_samples.loc[:, p2]

                # modified from:
                # https://matplotlib.org/stable/gallery/lines_bars_and_markers/
                # scatter_hist.html
                fig = Figure(figsize=(8, 8), dpi=dpi)

                # specify grid
                gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2),
                                      height_ratios=(2, 7), left=0.1,
                                      right=0.9, bottom=0.1, top=0.9,
                                      wspace=0.05, hspace=0.05)
                ax = fig.add_subplot(gs[1, 0])
                ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
                ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

                # draw scatter of the parameter pair
                ax.scatter(p1_samples, p2_samples)
                if param_names_on_plot:
                    ax.set_xlabel(param_names_on_plot[p1])
                    print(param_names_on_plot[p1])
                else:
                    ax.set_xlabel(p1)
                if param_names_on_plot:
                    ax.set_xlabel(param_names_on_plot[p2])
                    print(param_names_on_plot[p2])
                else:
                    ax.set_ylabel(p2)

                # draw histogram for each parameter on the side
                ax_histx.tick_params(axis='x', labelbottom=False)
                ax_histx.hist(p1_samples, bins=hist_bins)
                ax_histy.tick_params(axis='y', labelleft=False)
                ax_histy.hist(p2_samples, bins=hist_bins,
                              orientation='horizontal')

                fig.savefig(output_path)
                plt.close(fig)

    def plot_param_pairs(self, param_pairs, sessions=[], num_rows=4,
                         num_cols=2, page_size=(8.5, 11),
                         param_names_on_plot=None, titles=None):
        '''Make scatter plot of sample means for a pair of parameters, plus
        scatter plot of all samples of select sessions'''
        if not hasattr(self, 'sample_means'):
            print('No operation performed since sample means has not been'
                  + 'calculated')
            return

        num_subplots_per_page = num_rows * num_cols
        num_plots = 1
        if sessions:
            sessions = [s for s in sessions if s in self.session_list]
            num_plots += len(sessions)
        num_pages = math.ceil(num_plots / num_subplots_per_page)
        if param_names_on_plot:
            param_0_label = param_names_on_plot[param_pairs[0]]
            param_1_label = param_names_on_plot[param_pairs[1]]
        else:
            param_0_label = param_pairs[0]
            param_1_label = param_pairs[1]

        output_path = os.path.join(
            self.output_dir,
            f'param_pair_scatters_{param_pairs[0]}_{param_pairs[1]}.pdf')
        with PdfPages(output_path) as pdf:
            for page in range(num_pages):
                plt.figure(figsize=page_size, dpi=300)

                if page == num_pages - 1:
                    num_subplots = (num_plots - 1) % num_subplots_per_page + 1
                else:
                    num_subplots = num_subplots_per_page

                for plot_idx in range(num_subplots):
                    ax = plt.subplot(num_rows, num_cols, plot_idx + 1,
                                     box_aspect=1)

                    if page == 0 and plot_idx == 0:
                        # plot sample mean vs sample mean
                        session_orders = list(range(self.num_sessions))
                        ax.scatter(self.sample_means[param_pairs[0]],
                                   self.sample_means[param_pairs[1]],
                                   c=session_orders)
                        ax.set_xlabel(param_0_label)
                        ax.set_ylabel(param_1_label)
                        if titles:
                            ax.set_title(titles[0])
                        else:
                            ax.set_title('Sample means')
                        # plt.colorbar(sc, ax=ax, label='Cell positions')
                    else:
                        # plot sample vs sample in a session
                        plot_idx_all = page * num_subplots_per_page + plot_idx
                        session = sessions[plot_idx_all - 1]
                        session_idx = np.argwhere(
                            self.session_list == session)
                        if session_idx.size == 0:
                            continue
                        analyzer = self.session_analyzers[session_idx[0][0]]
                        samples = analyzer.get_samples()
                        ax.scatter(samples[param_pairs[0]],
                                   samples[param_pairs[1]], s=3, alpha=0.3)
                        ax.set_xlabel(param_0_label)
                        ax.set_ylabel(param_1_label)
                        if titles:
                            ax.set_title(titles[plot_idx_all])
                        else:
                            ax.set_title(session)

                plt.tight_layout()
                pdf.savefig()
                plt.close()

    def plot_rhats(self):
        """Make plots for R^hat for all parameters and log posterior

        Note that the R^hat is computed on all chains, including ones that are
        not mixed
        """
        if not isinstance(self.session_analyzers[0].raw_samples,
                          az.InferenceData):
            raise TypeError('Cannot plot R^hat if samples are not loaded '
                            + 'from Arviz\'s InferenceData')

        rhats = np.zeros((len(self.param_names) + 1, self.num_sessions))

        for i, analyzer in enumerate(self.session_analyzers):
            inf_data = analyzer.raw_samples

            # get R^hat for parameters
            param_rhats = az.rhat(inf_data, method='split')
            rhats[0, i] = param_rhats['sigma'].item()
            rhats[1:-1, i] = param_rhats['theta'].to_masked_array()

            # get R^hat for log posterior
            stat_rhats = az.rhat(inf_data.sample_stats, method='split')
            rhats[-1, i] = stat_rhats['lp'].item()

        # plot all R^hat's
        output_path = os.path.join(self.output_dir, 'rhats.pdf')
        pdf_multi_plot(plt.plot, rhats, output_path, '.', num_rows=4,
                       num_cols=1, titles=self.param_names + ['lp'],
                       xticks=self.session_list, xtick_rotation=90)

    def plot_posterior_rhats(self, dpi=300, xticks=None):
        '''Plot average R_hat of log posterior for mixed chains'''
        # get posterior R^hat's
        rhats = pd.DataFrame(columns=['R_hat'] + list(range(self.num_chains)))
        for analyzer in self.session_analyzers:
            mixed_chains, rhat = analyzer.get_mixed_chains(
                rhat_upper_bound=self.rhat_upper_bound, return_rhat=True)
            row = {c: int(c in mixed_chains)
                   for c in range(analyzer.num_chains)}
            row['R_hat'] = rhat
            rhats = rhats.append(row, ignore_index=True)

        # plot R^hat's
        output_path = os.path.join(self.output_dir, 'posterior_rhats.pdf')
        plt.figure(figsize=(11, 8.5), dpi=dpi)
        plt.plot(rhats['R_hat'], '.')
        if xticks is not None:
            plt.xticks(**xticks)
        plt.ylabel('R^hat')
        plt.ylim((0, self.rhat_upper_bound))
        plt.savefig(output_path)
        plt.close()

        # export R^hat's
        rhats.index = self.session_list
        output_path = os.path.join(self.output_dir, 'posterior_rhats.csv')
        rhats.to_csv(output_path)

    def plot_mean_lps_vs_trajectory_distances(self, ode, t0, ts, y0, y_ref,
                                              dpi=300, excluded_sessions=None):
        '''Make scatter plot of log posteriors vs trajectory distances'''
        # get R^hat's and mean trajectory distances
        mean_lps = []
        traj_dists = []
        for i, analyzer in enumerate(self.session_analyzers):
            if excluded_sessions and self.session_list[i] in excluded_sessions:
                continue

            lps = analyzer.get_mean_log_posteriors()
            mixed_chains = analyzer.get_mixed_chains(
                rhat_upper_bound=self.rhat_upper_bound)
            mixed_lps = [lps[i] for i in range(self.num_chains)
                         if i in mixed_chains]
            mean_lps.append(np.mean(mixed_lps))

            y0_cell = np.array([0, 0, 0.7, y0[i]])
            y_ref_cell = [None, None, None, y_ref[i, :]]
            traj_dists.append(analyzer.get_trajectory_distance(
                ode, t0, ts, y0_cell, y_ref_cell, 3, self.rhat_upper_bound))

        output_path = os.path.join(self.output_dir,
                                   'mean_log_posteriors_vs_traj_dists.pdf')
        plt.figure(figsize=(11, 8.5), dpi=dpi)
        plt.scatter(mean_lps, traj_dists)
        plt.xlabel('Mean log posterior')
        plt.ylabel('Mean trajectory distance')
        plt.savefig(output_path)
        plt.close()

        stats = {}
        stats['corr'], _ = scipy.stats.pearsonr(mean_lps, traj_dists)
        stats['r^2'] = np.corrcoef(mean_lps, traj_dists)[0, 1] ** 2

        return stats

    def plot_lp_rhats_vs_trajectory_distances(self, ode, t0, ts, y0, y_ref,
                                              dpi=300, excluded_sessions=None):
        '''Make scatter plot of R^hats of log posteriors vs trajectory
        distances
        '''
        # get R^hat's and mean trajectory_distances
        rhats = []
        traj_dists = []
        for i, analyzer in enumerate(self.session_analyzers):
            if excluded_sessions and self.session_list[i] in excluded_sessions:
                continue

            _, rhat = analyzer.get_mixed_chains(
                rhat_upper_bound=self.rhat_upper_bound, return_rhat=True)
            rhats.append(rhat)

            y0_cell = np.array([0, 0, 0.7, y0[i]])
            y_ref_cell = [None, None, None, y_ref[i, :]]
            traj_dists.append(analyzer.get_trajectory_distance(
                ode, t0, ts, y0_cell, y_ref_cell, 3, self.rhat_upper_bound))

        output_path = os.path.join(self.output_dir,
                                   'posterior_rhats_vs_traj_dists.pdf')
        plt.figure(figsize=(11, 8.5), dpi=dpi)
        plt.scatter(rhats, traj_dists)
        plt.xlabel('R^hat')
        plt.ylabel('Mean trajectory distance')
        plt.savefig(output_path)
        plt.close()

        stats = {}
        stats['corr'], _ = scipy.stats.pearsonr(rhats, traj_dists)
        stats['r^2'] = np.corrcoef(rhats, traj_dists)[0, 1] ** 2

        return stats

    def plot_mean_lps_vs_lp_rhats(self, dpi=300):
        '''Make scatter plot of log posteriors vs their R^hats'''
        mean_lps = []
        rhats = []

        for analyzer in self.session_analyzers:
            lps = analyzer.get_mean_log_posteriors()
            mixed_chains = analyzer.get_mixed_chains(
                rhat_upper_bound=self.rhat_upper_bound)
            mixed_lps = [lps[i] for i in range(self.num_chains)
                         if i in mixed_chains]
            mean_lps.append(np.mean(mixed_lps))

            _, rhat = analyzer.get_mixed_chains(
                rhat_upper_bound=self.rhat_upper_bound, return_rhat=True)
            rhats.append(rhat)

        output_path = os.path.join(
            self.output_dir, 'mean_log_posteriors_vs_posterior_rhats.pdf')
        plt.figure(figsize=(11, 8.5), dpi=dpi)
        plt.scatter(mean_lps, rhats)
        plt.xlabel('Mean log posterior')
        plt.ylabel('R^hat')
        plt.savefig(output_path)
        plt.close()

        stats = {}
        stats['corr'], _ = scipy.stats.pearsonr(mean_lps, rhats)
        stats['r^2'] = np.corrcoef(mean_lps, rhats)[0, 1] ** 2

        return stats

    def plot_sampling_time(self, time_unit='s', dpi=300, xticks=None,
                           hist_range=None):
        '''Plot runtime of all chains in every Stan sesssion'''
        time_unit_text = {'s': 'seconds', 'm': 'minutes', 'h': 'hours'}
        # load stan sample file
        sampling_time = []
        for analyzer in self.session_analyzers:
            sampling_time.append(analyzer.get_sampling_time(unit=time_unit))

        # plot sampling time as chain progression
        sampling_time = np.array(sampling_time)
        sampling_time_cap = 70000
        sampling_time = np.clip(sampling_time, None, sampling_time_cap)
        output_path = os.path.join(self.output_dir, 'sampling_time.pdf')
        plt.figure(figsize=(11, 8.5), dpi=dpi)
        for chain in range(self.num_chains):
            plt.plot(sampling_time[:, chain], color='C0', marker='.',
                     linestyle='')
        if xticks is not None:
            plt.xticks(**xticks)
        plt.ylabel(f'Time ({time_unit_text[time_unit]})')
        plt.savefig(output_path)
        plt.close()

        # plot histogram of sampling time
        output_path = os.path.join(self.output_dir, 'sampling_time_hist.pdf')
        plt.figure(figsize=(11, 8.5), dpi=dpi)
        plt.hist(sampling_time.flatten(), bins=20, range=hist_range)
        plt.savefig(output_path)
        plt.close()

        # export sampling time
        sampling_time = pd.DataFrame(sampling_time, index=self.session_list)
        sampling_time_table_path = os.path.join(self.output_dir,
                                                'sampling_time.csv')
        sampling_time.to_csv(sampling_time_table_path)

    def plot_mean_tree_depths(self, tree_depth_min=None, tree_depth_max=None,
                              dpi=300, xticks=None):
        '''Plot average tree depths during sampling'''
        # get mean tree depths
        tree_depths = np.array([analyzer.get_tree_depths().flatten()
                                for analyzer in self.session_analyzers])
        tree_depths = np.mean(tree_depths, axis=1)

        # plot mean tree depths
        output_path = os.path.join(self.output_dir, 'mean_tree_depths.pdf')
        plt.figure(figsize=(11, 8.5), dpi=dpi)
        plt.plot(tree_depths, '.')
        if xticks is not None:
            plt.xticks(**xticks)
        plt.ylim((tree_depth_min, tree_depth_max))  # set limits for the y-axis
        plt.ylabel('Mean tree depths')
        plt.savefig(output_path)
        plt.close()

        # export mean tree depths
        tree_depths = pd.Series(tree_depths)
        tree_depths.index = self.session_list
        output_path = os.path.join(self.output_dir, 'mean_tree_depths.csv')
        tree_depths.to_csv(output_path, header=False)

    def plot_mean_trajectory_distances(self, ode, t0, ts, y0, y_ref,
                                       dist_min=None, dist_max=None, dpi=300,
                                       xticks=None):
        traj_dists = []
        for i, analyzer in enumerate(self.session_analyzers):
            y0_cell = np.array([0, 0, 0.7, y0[i]])
            y_ref_cell = [None, None, None, y_ref[i, :]]
            traj_dists.append(analyzer.get_trajectory_distance(
                ode, t0, ts, y0_cell, y_ref_cell, 3, self.rhat_upper_bound))
        traj_dists = pd.Series(traj_dists)

        # plot mean trajectory distances
        output_path = os.path.join(self.output_dir,
                                   'mean_trajectory_distances.pdf')
        plt.figure(figsize=(11, 8.5), dpi=dpi)
        plt.plot(traj_dists, '.')
        if xticks is not None:
            plt.xticks(**xticks)
        plt.ylim((dist_min, dist_max))  # set limits for the y-axis
        plt.ylabel('Mean trajectory distances to input')
        plt.savefig(output_path)
        plt.close()

        # export trajectory distances
        traj_dists.index = self.session_list
        output_path = os.path.join(self.output_dir,
                                   'mean_trajectory_distances.csv')
        traj_dists.to_csv(output_path, header=False)

    def plot_mean_log_posteriors(self, dpi=300, xticks=None):
        """Plot mean posterior for each sampling chain"""
        # get mean log posterior
        mean_log_posteriors = []
        for analyzer in self.session_analyzers:
            mean_log_posteriors.append(analyzer.get_mean_log_posteriors())
        mean_log_posteriors = np.array(mean_log_posteriors)

        # plot mean log posterior
        output_path = os.path.join(self.output_dir, 'mean_log_posteriors.pdf')
        plt.figure(figsize=(11, 8.5), dpi=dpi)
        for chain in range(self.num_chains):
            plt.plot(mean_log_posteriors[:, chain], color='C0', marker='.',
                     linestyle='')
        if xticks is not None:
            plt.xticks(**xticks)
        plt.ylabel('Mean log posterior density')
        plt.savefig(output_path)
        plt.close()

        # save mean log posterior
        mean_lp_table_path = os.path.join(self.output_dir,
                                          'mean_log_posteriors.csv')
        mean_log_posteriors = pd.DataFrame(mean_log_posteriors,
                                           index=self.session_list)
        mean_log_posteriors.to_csv(mean_lp_table_path)

    def simulate_trajectories(self, ode, t0, ts, y0, y_ref, target_var_idx,
                              figure_size=(3, 2), dpi=300, output_paths=None,
                              exclude_sessions=[]):
        for idx, analyzer in enumerate(self.session_analyzers):
            if self.session_list[idx] in exclude_sessions:
                continue

            mixed_chains = analyzer.get_mixed_chains(
                rhat_upper_bound=self.rhat_upper_bound)

            if mixed_chains:
                y_sim = analyzer.simulate_chains(
                    ode, t0, ts, y0[idx, :], plot=False, verbose=False)
                for chain in mixed_chains:
                    if output_paths:
                        op = output_paths[idx][chain]
                    else:
                        op = os.path.join(
                            self.output_dir,
                            f'{idx:04d}_{self.session_list[idx]}_{chain}.pdf')

                    fig = Figure(figsize=figure_size, dpi=dpi)
                    ax = fig.gca()
                    ax.plot(ts, y_ref[idx, :], 'ko', fillstyle='none')
                    ax.plot(ts, y_sim[chain][:, :, target_var_idx].T,
                            color='C0', alpha=0.05)
                    fig.savefig(op)
                    plt.close(fig)

    def run_sensitivity_test(self, ode, t0, ts, y0, y_ref, target_var_idx,
                             test_params, method='default', method_kwargs=None,
                             plot_traj=False, figure_size=(3, 2), dpi=300,
                             figure_path_prefixes=None,
                             param_names_on_plot=None):
        '''Test sensitivity of parameters'''
        if method_kwargs is None:
            method_kwargs = {}

        test_percentiles = np.arange(0.1, 1, 0.1)  # percentiles for test
        num_test_percentiles = len(test_percentiles)
        test_quantiles = [0.01, 0.99]  # quantile to be test at
        num_steady_pts = ts.size // 5
        jet_cmap = plt.get_cmap('jet')
        colors = jet_cmap(test_percentiles)

        # initialize tables for stats
        stats = {}
        stats_types = ['MeanTrajDist', 'MeanPeakDiff', 'MeanPeakFoldChange',
                       'MeanSteadyDiff', 'MeanSteadyFoldChange']
        for param, qt in itertools.product(test_params, test_quantiles):
            stats[(param, qt)] = pd.DataFrame(columns=stats_types,
                                              index=self.session_list)

        def _test_quantile(q):
            test_value = session_samples[param].quantile(q)

            # initialize
            y_sim_all = np.empty((num_test_percentiles, ts.size))
            traj_dist = 0
            peak_diff = 0
            steady_diff = 0
            peak_fold_change = 0
            steady_fold_change = 0
            peak_ref = np.max(y_ref[idx, :])
            steady_ref = np.mean(y_ref[idx, -num_steady_pts:])

            for i, pct in enumerate(test_percentiles):
                row_cond = param_samples == \
                    param_samples.quantile(pct, interpolation='nearest')
                row_idx = param_samples[row_cond].index[0]
                if method == 'mode':
                    theta = sample_modes.copy()
                else:
                    theta = session_samples.iloc[row_idx, 1:].copy()
                theta[param] = test_value

                y_sim = simulate_trajectory(ode, theta, t0, ts, y0[idx, :])
                y_sim = y_sim[:, target_var_idx]

                # get trajectory distance
                traj_dist += np.linalg.norm(y_sim - y_ref[idx, :])

                # get stats for change in peak
                peak_sim = np.max(y_sim)
                peak_diff += peak_sim - peak_ref
                if peak_ref > peak_sim:  # negative fold change
                    peak_fold_change -= peak_ref / peak_sim
                else:  # positive fold change
                    peak_fold_change += peak_sim / peak_ref

                # get stats for change in steady state
                steady_sim = np.mean(y_sim[-num_steady_pts:])
                steady_diff += steady_sim - steady_ref
                if steady_ref > steady_sim:  # negative fold change
                    steady_fold_change -= steady_ref / steady_sim
                else:  # positive fold change
                    steady_fold_change += steady_sim / steady_ref

                if plot_traj:
                    y_sim_all[i, :] = y_sim

            if plot_traj:
                if figure_path_prefixes:
                    figure_path = f'{figure_path_prefixes[idx]}_{param}_' \
                        + f'{q:.2f}.pdf'
                else:
                    figure_path = f'{idx:04d}_{session}_{param}_{q:.2f}.pdf'
                figure_path = os.path.join(self.output_dir, figure_path)
                fig = Figure(figsize=figure_size, dpi=dpi)
                ax = fig.gca()

                for i, pct in enumerate(test_percentiles):
                    ax.plot(ts, y_sim_all[i, :], color=colors[i],
                            label=f'{pct:.1f}')

                ax.plot(ts, y_ref[idx, :], 'ko', fillstyle='none')
                if param_names_on_plot:
                    param_name = param_names_on_plot[param]
                else:
                    param_name = param
                ax.set_title(f'{param_name}, {q}')
                ax.legend()
                fig.savefig(figure_path)
                plt.close(fig)

            # record all stats
            stats[(param, q)].loc[session, 'MeanTrajDist'] = \
                traj_dist / num_test_percentiles
            stats[(param, q)].loc[session, 'MeanPeakDiff'] = \
                peak_diff / num_test_percentiles
            stats[(param, q)].loc[session, 'MeanPeakFoldChange'] = \
                peak_fold_change / num_test_percentiles
            stats[(param, q)].loc[session, 'MeanSteadyDiff'] = \
                steady_diff / num_test_percentiles
            stats[(param, q)].loc[session, 'MeanSteadyFoldChange'] = \
                steady_fold_change / num_test_percentiles

        for idx, analyzer in enumerate(self.session_analyzers):
            session = self.session_list[idx]
            session_samples = analyzer.get_samples()
            if method == 'mode':
                sample_modes = analyzer.get_sample_modes(**method_kwargs)

            for param in test_params:
                param_samples = session_samples[param]
                for qt in test_quantiles:
                    _test_quantile(qt)

        # save tables for stats
        for param, qt in itertools.product(test_params, test_quantiles):
            stats_path = os.path.join(self.output_dir,
                                      f'{param}_{qt}_stats.csv')
            stats[(param, qt)].to_csv(stats_path)

    def load_expression_data(self, data_path, use_highly_variable_genes=False):
        """Load and preprocess expression data"""
        # load raw expression data
        self.raw_data = pd.read_csv(data_path, sep='\t')

        # perform log normalization
        if use_highly_variable_genes:
            adata = AnnData(self.raw_data.T)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata)
            self.gene_symbols = adata.var_names[adata.var['highly_variable']]
            self.log_data = np.log1p(
                self.raw_data.loc[self.gene_symbols, :].to_numpy())
        else:
            self.log_data = np.log1p(self.raw_data.to_numpy())
            self.gene_symbols = self.raw_data.index.to_numpy()

        # change type of session list to int
        self.session_list = self.session_list.astype(int)

        # plot sample means vs genes expression if the latter is available
        if hasattr(self, 'sample_means'):
            gvp_dir = os.path.join(self.output_dir, 'genes-vs-params')
            if not os.path.exists(gvp_dir):
                os.mkdir(gvp_dir)
            session_orders = list(range(self.num_sessions))

            for gene_idx, gene in enumerate(self.gene_symbols):
                gvp_scatter_path = os.path.join(gvp_dir, f'{gene}.pdf')
                self._scatter_multi_plot(
                    self.log_data[gene_idx, self.session_list],
                    gvp_scatter_path, c=session_orders)

    def run_pca(self, num_pcs=50, num_top_pcs=10, sampled_only=False,
                plot=False):
        """Perform PCA"""
        self.num_pcs = num_pcs
        self.num_top_pcs = num_top_pcs
        self.pca = PCA(n_components=self.num_pcs)
        if sampled_only:
            self.log_pca_data = self.pca.fit_transform(
                self.log_data[:, self.session_list].T)
        else:
            self.log_pca_data = self.pca.fit_transform(self.log_data.T)
        self.log_pca_data_abs = np.abs(self.log_pca_data)

        # plot PCA explained variance ratio
        if plot:
            plt.figure(figsize=(11, 8.5))
            plt.plot(self.pca.explained_variance_ratio_, '.')
            plt.savefig(os.path.join(self.output_dir, 'pca_var_ratio.pdf'))
            plt.close()

    def get_top_genes_from_pca(self, num_top_genes=10, plot=False):
        pca_loadings = self.pca.components_.T
        top_pos_genes = pd.DataFrame(index=range(num_top_genes),
                                     columns=range(self.num_pcs))
        top_neg_genes = pd.DataFrame(index=range(num_top_genes),
                                     columns=range(self.num_pcs))

        top_gene_dir = os.path.join(self.output_dir, 'pca-top-genes')
        if not os.path.exists(top_gene_dir):
            os.mkdir(top_gene_dir)

        for comp in range(self.num_pcs):
            ranked_gene_indices = np.argsort(pca_loadings[:, comp])
            top_pos_genes_comp = self.gene_symbols[
                ranked_gene_indices[:-num_top_genes-1:-1]]
            top_pos_genes.loc[:, comp] = top_pos_genes_comp
            top_neg_genes_comp = self.gene_symbols[
                ranked_gene_indices[:num_top_genes]]
            top_neg_genes.loc[:, comp] = top_neg_genes_comp

            if plot and comp < self.num_top_pcs:
                # plot top positive genes vs sampled means
                for i, gene in enumerate(top_pos_genes_comp):
                    gene_idx = ranked_gene_indices[-(i+1)]
                    gene_log_data = self.log_data[gene_idx, self.session_list]
                    gene_scatter_path = os.path.join(
                        top_gene_dir,
                        f'comp_{comp}_top_pos_{i:02d}_{gene}_vs_params.pdf')
                    self._scatter_multi_plot(gene_log_data, gene_scatter_path)

                # plot top negative genes vs sampled means
                for i, gene in enumerate(top_neg_genes_comp):
                    gene_idx = ranked_gene_indices[i]
                    gene_log_data = self.log_data[gene_idx, self.session_list]
                    gene_scatter_path = os.path.join(
                        top_gene_dir,
                        f'comp_{comp}_top_neg_{i:02d}_{gene}_vs_params.pdf')
                    self._scatter_multi_plot(gene_log_data, gene_scatter_path)
        # save top genes
        top_pos_genes.to_csv(os.path.join(top_gene_dir, 'top_pos_genes.csv'))
        top_neg_genes.to_csv(os.path.join(top_gene_dir, 'top_neg_genes.csv'))

        self.top_pos_genes = top_pos_genes
        self.top_neg_genes = top_neg_genes
        top_pos_gene_list = np.unique(
            self.top_pos_genes.loc[:, :self.num_top_pcs - 1])
        top_neg_gene_list = np.unique(
            self.top_neg_genes.loc[:, :self.num_top_pcs - 1])
        top_pc_gene_list = np.unique(
            np.concatenate((top_pos_gene_list, top_neg_gene_list), axis=None))
        self.top_pos_gene_list = top_pos_gene_list
        self.top_neg_gene_list = top_neg_gene_list
        self.top_pc_gene_list = top_pc_gene_list

    def compute_gene_param_correlations(self, genes, plot=False):
        '''Compute correlations between parameters and top genes from PCA'''
        gvp_corr_dir = os.path.join(self.output_dir, 'genes-vs-params')
        if not os.path.exists(gvp_corr_dir):
            os.mkdir(gvp_corr_dir)

        gene_vs_param_corrs = pd.DataFrame(
            columns=self.param_names, index=genes)
        gene_vs_param_corr_pvals = pd.DataFrame(
            columns=self.param_names, index=genes)

        for gene, param in itertools.product(genes, self.param_names):
            gene_idx = np.where(gene == self.gene_symbols)[0][0]
            corr, p_val = scipy.stats.pearsonr(
                self.log_data[gene_idx, self.session_list],
                self.sample_means[param])
            gene_vs_param_corrs.loc[gene, param] = corr
            gene_vs_param_corr_pvals.loc[gene, param] = p_val

        # save Pearson correlations
        gene_vs_param_corrs.to_csv(
            os.path.join(gvp_corr_dir, 'pearson_corrs.csv'))
        gene_vs_param_corr_pvals.to_csv(
            os.path.join(gvp_corr_dir, 'pearson_corrs_p_vals.csv'))
        self.gene_vs_param_corrs = gene_vs_param_corrs
        self.gene_vs_param_corr_pvals = gene_vs_param_corr_pvals

        if plot:
            for param in self.param_names:
                plt.hist(gene_vs_param_corrs[param], bins=15)
                figure_path = os.path.join(gvp_corr_dir,
                                           f'gene_vs_{param}_corr_hist.png')
                plt.savefig(figure_path)
                plt.close()

        # sort gene-parameter pairs in descending order of Pearson correlations
        num_gene_param_pairs = len(genes) * self.num_params
        sorted_gene_vs_param_pairs = pd.DataFrame(
            columns=['Gene', 'Parameter', 'Correlation', 'p-value'],
            index=range(num_gene_param_pairs))

        for i, (gene, param) in enumerate(
                itertools.product(genes, self.param_names)):
            corr = gene_vs_param_corrs.loc[gene, param]
            p_val = gene_vs_param_corr_pvals.loc[gene, param]
            sorted_gene_vs_param_pairs.iloc[i, :] = [gene, param, corr, p_val]

        sorted_gene_vs_param_pairs.sort_values(
            'Correlation', ascending=False, inplace=True, ignore_index=True,
            key=lambda x: np.abs(x))

        # save sorted gene-parameter pairs
        sorted_gene_vs_param_pairs.to_csv(
            os.path.join(gvp_corr_dir, 'pearson_corrs_sorted.csv'))
        self.sorted_gene_vs_param_pairs = sorted_gene_vs_param_pairs

    def run_genes_vs_params_regression(self, regressor_name, genes, degree=1,
                                       select_pairs=[], plot=False):
        '''Fit a regression model for gene expression vs sampled means of
        params, with features raised to a specified degree'''
        regression_dir = os.path.join(self.output_dir,
                                      'genes-vs-params-regression')
        if not os.path.exists(regression_dir):
            os.mkdir(regression_dir)

        regressor_classes = {
            'linear': LinearRegression, 'huber': HuberRegressor,
            'gamma': GammaRegressor}
        regressors_trained = {}
        r2_scores = pd.DataFrame(index=genes, columns=self.param_names)
        mean_sq_errors = pd.DataFrame(index=genes, columns=self.param_names)
        for gene in genes:
            param_regressors = {}
            # get expression data, of shape num_cells * 1
            gene_idx = np.where(self.gene_symbols == gene)[0][0]
            X_gene = self.log_data[gene_idx, self.session_list, np.newaxis]
            # generate features from expression data
            if degree > 1:
                poly = PolynomialFeatures(degree)
                X = poly.fit_transform(X_gene)
            else:
                X = X_gene

            # perform regression for each param
            for param in self.param_names:
                regressor = regressor_classes[regressor_name]()
                y = self.sample_means[param]

                regressor.fit(X, y)
                y_pred = regressor.predict(X)
                param_regressors[param] = regressor

                # compute metrics
                r2_scores.loc[gene, param] = regressor.score(X, y)
                mean_sq_errors.loc[gene, param] = mean_squared_error(y, y_pred)

                # save regressor
                if (gene, param) in select_pairs:
                    regressors_trained[
                        (degree, gene, param)] = regressor

            # plot regression lines (curves)
            if plot:
                regression_scatter_path = os.path.join(
                    regression_dir,
                    f'{regressor_name}_degree_{degree}_{gene}.pdf')
                self._scatter_multi_plot(X_gene, regression_scatter_path,
                                         regressors=param_regressors, X_poly=X)

        # save metrics
        r2_scores_path = os.path.join(
            regression_dir, f'{regressor_name}_degree_{degree}_scores.csv')
        r2_scores.to_csv(r2_scores_path, float_format='%.8f')

        mean_sq_errors_path = os.path.join(
            regression_dir, f'{regressor_name}_degree_{degree}_mse.csv')
        mean_sq_errors.to_csv(mean_sq_errors_path, float_format='%.8f')

        return regressors_trained

    def _scatter_multi_plot(self, X_data, output_path, c=None, num_rows=4,
                            num_cols=2, regressors=None, X_poly=None):
        """Make multiple scatter plots in a PDF"""
        num_subplots_per_page = num_rows * num_cols
        num_plots = self.num_params
        num_pages = math.ceil(num_plots / num_subplots_per_page)

        with PdfPages(output_path) as pdf:
            # generate each page
            for page in range(num_pages):
                # set page size as US letter
                plt.figure(figsize=(8.5, 11))

                # set number of subplots in current page
                if page == num_pages - 1:
                    num_subplots = (num_plots - 1) % num_subplots_per_page + 1
                else:
                    num_subplots = num_subplots_per_page

                # make each subplot
                for plot_idx in range(num_subplots):
                    plt.subplot(num_rows, num_cols, plot_idx + 1)
                    param = self.param_names[
                        page * num_subplots_per_page + plot_idx]
                    if isinstance(c, dict) or isinstance(c, pd.DataFrame):
                        plt.scatter(X_data, self.sample_means[param],
                                    c=c[param])
                    else:
                        plt.scatter(X_data, self.sample_means[param], c=c)
                    plt.title(param)
                    if c is not None:
                        plt.colorbar()

                    # plot regression line/curve
                    if regressors:
                        sample_mean_pred = regressors[param].predict(X_poly)
                        plt.scatter(X_data, sample_mean_pred)

                plt.tight_layout()
                pdf.savefig()
                plt.close()

    def plot_select_genes_vs_params(self, select_pairs, regressors_trained,
                                    figure_name, figure_size=(8.5, 11),
                                    num_rows=4, num_cols=2, degree=1,
                                    show_corrs=True, param_names_on_plot=None,
                                    **scatter_kwargs):
        """Plot select gene-parameter pairs with regression line/curve"""
        num_subplots_per_page = num_rows * num_cols
        num_plots = len(select_pairs)
        num_pages = math.ceil(num_plots / num_subplots_per_page)
        cell_orders = np.arange(self.session_list.size)

        figure_path = os.path.join(self.output_dir, figure_name)
        with PdfPages(figure_path) as pdf:
            # generate each page
            for page in range(num_pages):
                # set page size as US letter
                plt.figure(figsize=figure_size)

                # set number of subplots in current page
                if page == num_pages - 1:
                    num_subplots = (num_plots - 1) % num_subplots_per_page + 1
                else:
                    num_subplots = num_subplots_per_page

                # make each subplot
                for plot_idx in range(num_subplots):
                    plt.subplot(num_rows, num_cols, plot_idx + 1)

                    # get data
                    gene, param = select_pairs[
                        page * num_subplots_per_page + plot_idx]
                    gene_idx = np.where(gene == self.gene_symbols)[0][0]
                    X_data = self.log_data[gene_idx, self.session_list,
                                           np.newaxis]

                    # make scatter plot
                    plt.scatter(X_data, self.sample_means[param],
                                c=cell_orders, **scatter_kwargs)
                    if param_names_on_plot:
                        param_name = param_names_on_plot[param]
                    else:
                        param_name = param
                    if show_corrs:
                        corr = self.gene_vs_param_corrs.loc[gene, param]
                        plt.title(f'{gene} vs {param_name}: {corr:.6f}')
                    else:
                        plt.title(f'{gene} vs {param_name}')

                    # plot regression line/curve
                    if degree == 1:
                        X = X_data
                    else:
                        poly = PolynomialFeatures(degree)
                        X = poly.fit_transform(X_data)
                    regressor = regressors_trained[(degree, gene, param)]
                    sample_mean_pred = regressor.predict(X)
                    plt.scatter(X_data, sample_mean_pred, c='C1',
                                **scatter_kwargs)

                plt.tight_layout()
                pdf.savefig()
                plt.close()

# utility functions
def load_trajectories(t0, filter_type=None, moving_average_window=20,
                      downsample_offset=-1, downsample_factor=10,
                      verbose=False):
    """Preprocess raw trajectories with filter and downsampling"""
    if verbose:
        print('Loading calcium trajectories...')
    y = np.loadtxt('canorm_tracjectories.csv', delimiter=',')

    # filter the trajectories
    if filter_type == 'moving_average':
        y = moving_average(y, window=moving_average_window, verbose=verbose)
    elif filter_type == 'savitzky_golay':
        from scipy.signal import savgol_filter
        y = savgol_filter(y, 51, 2)
    elif filter_type is not None and verbose:
        print(f'Unsupported filter {filter_type}. The trajectories will not '
              + 'be filtered.')

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

def get_trajectory_derivatives(t0, downsample_offset=-1, downsample_factor=10,
                               verbose=False):
    '''Compute first derivatives of trajectories'''
    from scipy.signal import savgol_filter

    # load trajectories
    if verbose:
        print('Loading calcium trajectories...')
    y = np.loadtxt('canorm_tracjectories.csv', delimiter=',')

    # smooth trajectories
    y = savgol_filter(y, 51, 2)

    # get derivatives
    dydt = np.gradient(y, axis=1)

    # downsample the trajectories
    t_end = y.shape[1] - 1
    if downsample_offset >= 0 and downsample_factor > 1:
        ts = np.concatenate((
            np.arange(t0 + 1, downsample_offset),
            np.arange(downsample_offset, t_end + 1, downsample_factor)
        ))
    else:
        ts = np.arange(t0 + 1, t_end + 1)

    dydt = dydt[:, ts]
    ts -= t0

    return dydt, ts

def moving_average(x: np.ndarray, window: int = 20, verbose=True):
    """Compute moving average of trajectories"""
    if verbose:
        print(f'Performing moving average with window size of {window}...')
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
        sample_path = os.path.join(sample_dir, f'chain_{chain_idx}.csv')

        # get number of warm-up iterations
        num_warmup_iters = 0
        if not include_warmup_iters:
            with open(sample_path, 'r') as sf:
                for line in sf:
                    if 'warmup=' in line:
                        num_warmup_iters = int(line.strip().split('=')[-1])
                        break

        # get samples
        chain_samples = pd.read_csv(sample_path, index_col=False, comment='#')
        stan_samples.append(chain_samples.iloc[num_warmup_iters:, 7:])

    return stan_samples

def load_stan_fit_samples(sample_file, include_warmup_iters=False):
    """Load samples output by StanFit4Model"""
    stan_samples = []
    raw_samples = pd.read_csv(sample_file, index_col=0)
    chains = np.unique(raw_samples['chain'])

    # add samples for each chain
    for chain_idx in chains:
        samples = raw_samples.loc[raw_samples['chain'] == chain_idx, :]
        # remove rows for warmup iterations
        if not include_warmup_iters:
            samples = samples.loc[samples['warmup'] == 0, :]
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
    num_chains = inf_data.sample_stats.dims['chain']
    sigma_array = inf_data.posterior['sigma'].values
    theta_array = inf_data.posterior['theta'].values

    # gather samples for each chain
    for chain_idx in range(num_chains):
        samples = np.concatenate((sigma_array[chain_idx, :, np.newaxis],
                                  theta_array[chain_idx, :, :]), axis=1)
        stan_samples.append(pd.DataFrame(samples))

    return stan_samples

def get_prior_from_samples(prior_dir, prior_chains,
                           sample_source='arviz_inf_data', verbose=True):
    """Get prior distribution from sampled parameters"""
    if verbose:
        chain_str = ', '.join(map(str, prior_chains))
        print(f'Getting prior distribution from chain(s) {chain_str}...')
        sys.stdout.flush()

    # load sampled parameters
    if sample_source == 'sample_files':
        stan_samples = load_stan_sample_files(prior_dir, max(prior_chains) + 1)
    elif sample_source == 'fit_export':
        fit_sample_file = os.path.join(prior_dir, 'stan_fit_samples.csv')
        stan_samples = load_stan_fit_samples(fit_sample_file)
    else:  # sample_source == Arviz InferenceData
        inf_data_file = os.path.join(prior_dir, 'arviz_inf_data.nc')
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

def simulate_trajectory(ode, theta, t0, ts, y0, integrator='dopri5',
                        **integrator_params):
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

def pdf_multi_plot(plot_func, plot_data, output_path, *args, num_rows=4,
                   num_cols=2, page_size=(8.5, 11), dpi=300, titles=None,
                   xticks=None, xtick_pos=None, xtick_rotation=0,
                   show_progress=False):
    """make multiple plots in a PDF"""
    num_subplots_per_page = num_rows * num_cols
    num_plots = len(plot_data)
    num_pages = math.ceil(num_plots / num_subplots_per_page)
    if xticks is not None and xtick_pos is None:
        xtick_pos = np.arange(len(xticks))

    with PdfPages(output_path) as pdf:
        # generate each page
        for page in tqdm(range(num_pages), disable=not show_progress):
            # set page size as US letter
            plt.figure(figsize=page_size, dpi=dpi)

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

def get_mode_continuous_rv(x, method='kde', bins=100):
    mode = 0

    if method == 'histogram':
        hist, bin_edges = np.histogram(x, bins=bins)
        max_bin = np.argmax(hist)
        mode = (bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2
    else:  # method == 'kde'
        kde = scipy.stats.gaussian_kde(x)
        support = np.linspace(np.amin(x), np.amax(x), 100)
        pdf = kde.evaluate(support)
        mode = support[np.argmax(pdf)]

    return mode

def get_kl_nn(posterior_samples, random_seed=0, verbose=False):
    '''Compute KL divergence based nearest neighbors

    samples: list of posterior samples, each has shape num_draws * num_params
    See https://github.com/wollmanlab/ODEparamFitting_ABCSMC/blob/master/estimateKLdivergenceBasedOnNN.m
    '''
    from sklearn.neighbors import NearestNeighbors
    bit_generator = np.random.MT19937(random_seed)
    rng = np.random.default_rng(bit_generator)

    k = 2
    num_samples = len(posterior_samples)
    D = np.ones((num_samples, num_samples))
    nn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree')

    for i in range(num_samples):
        D[i, i] = 0.5

        for j in range(i + 1, num_samples):
            if verbose:
                print(f'\rFinding nearest neighbors for samples {i} and {j}',
                      end='', flush=False)

            n_i = posterior_samples[i].shape[0]
            n_j = posterior_samples[j].shape[0]
            n = min(n_i, n_j)
            P_i = posterior_samples[i][rng.choice(n_i, size=n), :]
            P_j = posterior_samples[j][rng.choice(n_j, size=n), :]
            P_ij = np.vstack((P_i, P_j))

            nn.fit(P_ij)
            _, nn_indices = nn.kneighbors(P_ij)
            nn_indices = nn_indices[:, k - 1]
            D[i, j] = np.mean(np.concatenate(
                ((nn_indices[:n] < n).astype(int),
                 (nn_indices[n:] >= n).astype(int))
            ))

    if verbose:
        print()

    D = 1 - D.T
    KL = D * np.log(D * 2) + (1 - D) * np.log((1 - D) * 2)
    for i, j in itertools.combinations(range(num_samples), 2):
        if np.isnan(KL[j, i]):
            KL[j, i] = 1.0

        KL[i, j] = KL[j, i]

    return KL

def get_jensen_shannon(posterior_samples, subsample_size=1000, random_seed=0,
                       verbose=False):
    '''Compute Jensen-Shannon distance between pairs of samples'''
    bit_generator = np.random.MT19937(random_seed)
    rng = np.random.default_rng(bit_generator)

    num_samples = len(posterior_samples)
    num_params = posterior_samples[0].shape[1]
    js_dists = np.empty((num_samples, num_samples))

    for i, j in itertools.combinations_with_replacement(range(num_samples), 2):
        if verbose:
            print('Computing Jensen-Shannon distance between sample '
                  f'{i:04d} and sample {j:04d}...')

        sample_i = posterior_samples[i]
        sample_j = posterior_samples[j]
        sample_min = np.minimum(np.amin(sample_i, axis=0),
                                np.amin(sample_j, axis=0))
        sample_max = np.maximum(np.amax(sample_i, axis=0),
                                np.amax(sample_j, axis=0))
        estimation_points = \
            rng.random(size=(subsample_size, num_params)) \
                * (sample_max - sample_min) + sample_min

        kernel_i = scipy.stats.gaussian_kde(sample_i.T)
        density_i = kernel_i(estimation_points.T)
        kernel_j = scipy.stats.gaussian_kde(sample_j.T)
        density_j = kernel_j(estimation_points.T)

        js_ij =  scipy.spatial.distance.jensenshannon(density_i, density_j)
        if np.isnan(js_ij):
            js_dists[i, j] = js_dists[j, i] = 1.0
        else:
            js_dists[i, j] = js_dists[j, i] = js_ij

    return js_dists
