# %%
import os
import os.path
import itertools
import math
import json
import numpy as np
import scipy.io
import scipy.stats
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm

from stan_helpers import StanSessionAnalyzer, StanMultiSessionAnalyzer, \
    load_trajectories, get_trajectory_derivatives, simulate_trajectory, \
    get_mode_continuous_rv
import calcium_models

working_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(working_dir)

# %%
# initialize cell chain analysis

# specify a cell chain
# stan_run = '1'
# stan_run = '2'
# stan_run = '3'
# stan_run = '3-1.0'
# stan_run = '3-2.0'
# stan_run = 'simple-prior'
# stan_run = 'const-eta1'
# stan_run = 'const-Be'
stan_run = 'const-Be-eta1'
# stan_run = 'const-Be-eta1-mixed-2'
# stan_run = 'lemon-prior-1000'
# stan_run = 'lemon-prior-500'

# additional flags
first_cell_order = 1
last_cell_order = 500
sensitive_params = ['L', 'Katp', 'KoffPLC', 'Vplc', 'Kip3', 'KoffIP3', 'a',
                    'dinh', 'd1', 'epr', 'eta2']
pca_sampled_only = False
use_custom_xticks = True

# load metadata
with open('stan_run_meta.json', 'r') as f:
    stan_run_meta = json.load(f)

# get parameter names
param_mask = stan_run_meta[stan_run]['param_mask']
param_names = [calcium_models.param_names[i + 1]
               for i, mask in enumerate(param_mask) if mask == "1"]
param_names = ['sigma'] + param_names
num_params = len(param_names)
# select_params = 'd5', 'eta2', 'KoffIP3', 'k3', 'epr', 'KoffPLC', 'Katp'
select_param_pairs = [('KoffPLC', 'Katp'), ('eta3', 'c0'), ('epr', 'eta2'),
                      ('a', 'dinh'), ('KoffPLC', 'a')]

# get cell list
cell_list_path = os.path.join(
    'cell_lists', stan_run_meta[stan_run]['cell_list'])
full_cell_list = pd.read_csv(cell_list_path, sep='\t')
cell_list = full_cell_list.iloc[first_cell_order:last_cell_order + 1, :]

# get directories for sampled cells, as well as output of analysis
output_root = os.path.join(
    '../../result', stan_run_meta[stan_run]['output_dir'])
output_dir = \
    f'multi-sample-analysis-{first_cell_order:04d}-{last_cell_order:04d}'
if pca_sampled_only:
    output_dir += '-pca-sampled-only'
output_dir = os.path.join(output_root, output_dir)
session_list = [str(c) for c in cell_list['Cell']]
session_dirs = [f'samples/cell-{c:04d}' for c in cell_list['Cell']]
session_dirs = [os.path.join(output_root, sd) for sd in session_dirs]

# load calcium trajectories
ode_variant = stan_run_meta[stan_run]['ode_variant']
calcium_ode = getattr(calcium_models, f'calcium_ode_{ode_variant}')
t0 = 200
t_downsample = 300
y_all, y0_all, ts = load_trajectories(t0, filter_type='moving_average',
    moving_average_window=20, downsample_offset=t_downsample)
y_prime, _ = get_trajectory_derivatives(t0, downsample_offset=t_downsample)

# get similarity matrix
soptsc_vars = scipy.io.loadmat(
        '../../result/SoptSC/SoptSC_feature_100/workspace.mat')
similarity_matrix = soptsc_vars['W']

# initialize the analyzer for the cell chain
print('Initializing the analyzer for the cell chain...')
analyzer = StanMultiSessionAnalyzer(session_list, output_dir, session_dirs,
                                    param_names=param_names)
mixed_cells = [int(c) for c in analyzer.session_list]
num_mixed_cells = analyzer.num_sessions

# initialize the analyzer for root cell
print('Initializing the analyzer for root cell...')
root_output_dir = os.path.join(
    '../../result', stan_run_meta[stan_run]['root_cell_dir'])
root_analyzer = StanSessionAnalyzer(root_output_dir, param_names=param_names)

# change font settings
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['font.size'] = 16

# set ticks on x-axis for plots
if use_custom_xticks:
    if num_mixed_cells > 50:
        num_xticks = int(np.round(num_mixed_cells / 20)) + 1
        xtick_locs = np.arange(num_xticks) * 20 - 1
        xtick_locs[0] += 1
        xtick_labels = xtick_locs + first_cell_order
    elif num_mixed_cells > 10:
        num_xticks = int(np.round(num_mixed_cells / 5)) + 1
        xtick_locs = np.arange(num_xticks) * 5
        xtick_labels = xtick_locs + first_cell_order
    else:
        num_xticks = num_mixed_cells
        xtick_locs = np.arange(num_xticks)
        xtick_labels = xtick_locs + first_cell_order

    xticks = {'ticks': xtick_locs, 'labels': xtick_labels}
else:
    xticks = None

# %%
# simulate trajectories using sampled parameters
traj_dir = os.path.join(output_root, 'sampled-trajectories')
if not os.path.exists(traj_dir):
    os.mkdir(traj_dir)

figure_paths = []
for i, cell in enumerate(mixed_cells):
    cell_fps = []
    cell_order = first_cell_order + i
    for chain in range(analyzer.num_chains):
        chain_fp = os.path.join(
            traj_dir, f'{cell_order:04d}_cell_{cell}_{chain}.pdf')
        cell_fps.append(chain_fp)
    figure_paths.append(cell_fps)

y0_run = np.zeros((num_mixed_cells, 4))
y0_run[:, 2] = 0.7
y0_run[:, 3] = y0_all[mixed_cells]
y_run = y_all[mixed_cells, :]
analyzer.simulate_trajectories(calcium_ode, 0, ts, y0_run, y_run, 3,
                               output_paths=figure_paths)

# %%
from matplotlib.backends.backend_pdf import PdfPages

def plot_trajectories_pdf(y, figure_path, titles=None, num_rows=4, num_cols=2):
    num_subplots_per_page = num_rows * num_cols
    num_plots = num_mixed_cells
    num_pages = math.ceil(num_plots / num_subplots_per_page)

    with PdfPages(figure_path) as pdf:
        # generate each page
        for page in range(num_pages):
            # set page size as US letter
            plt.figure(figsize=(8.5, 11), dpi=300)

            # set number of subplots in current page
            if page == num_pages - 1:
                num_subplots = (num_plots - 1) % num_subplots_per_page + 1
            else:
                num_subplots = num_subplots_per_page

            # make each subplot
            for plot_idx in range(num_subplots):
                y_row = page * num_subplots_per_page + plot_idx
                plt.subplot(num_rows, num_cols, plot_idx + 1)
                plt.plot(ts, y[y_row, :])

                if titles:
                    plt.title(mixed_cells[y_row])

            plt.tight_layout()
            pdf.savefig()
            plt.close()

# %%
# simulate trajectories using Lemon prior
lemon_prior_spec_path = os.path.join('stan_models', ode_variant,
                                     'calcium_model_alt_prior.txt')
lemon_prior_spec = pd.read_csv(lemon_prior_spec_path, delimiter='\t',
                               index_col=0)
lemon_prior = lemon_prior_spec['mu'].to_numpy()

y_lemon = np.empty((num_mixed_cells, ts.size))
for i, cell in enumerate(mixed_cells):
    y_cell = simulate_trajectory(calcium_ode, lemon_prior, 0, ts,
                                 np.array([0, 0, 0.7, y0_all[cell]]))
    y_lemon[i, :] = y_cell[:, 3]

lemon_traj_path = os.path.join(
    '../../result', 'trajectories',
    'dfs_feature_100_root_5106_0.000_1.8_lemon_prior.pdf')
plot_trajectories_pdf(y_lemon, lemon_traj_path, titles=mixed_cells)

# %%
# simulate trajectories using mode
analyzer.get_sample_modes(method='histrogram')

y_mode = np.empty((num_mixed_cells, ts.size))
for i, cell in enumerate(mixed_cells):
    thetas = analyzer.sample_modes.iloc[i, 1:].to_numpy()
    y_cell = simulate_trajectory(calcium_ode, thetas, 0, ts,
                                 np.array([0, 0, 0.7, y0_all[cell]]))
    y_mode[i, :] = y_cell[:, 3]

mode_traj_path = os.path.join(output_dir, 'mode_trajectories.pdf')
plot_trajectories_pdf(y_mode, mode_traj_path, titles=mixed_cells)

# %%
# sensitivity test
use_mode = False
sensitivity_dir = 'sensitivity-test'
if use_mode:
    sensitivity_dir += '-mode'
sensitivity_dir = os.path.join(output_root, sensitivity_dir)
if not os.path.exists(sensitivity_dir):
    os.mkdir(sensitivity_dir)
# temporarily change output directory
analyzer.output_dir = sensitivity_dir

figure_paths = []
for i, cell in enumerate(mixed_cells):
    cell_order = first_cell_order + i
    figure_paths.append(f'{cell_order:04d}_cell_{cell}')

y0_run = np.zeros((num_mixed_cells, 4))
y0_run[:, 2] = 0.7
y0_run[:, 3] = y0_all[mixed_cells]
y_run = y_all[mixed_cells, :]

if use_mode:
    analyzer.run_sensitivity_test(
        calcium_ode, 0, ts, y0_run, y_run, 3, sensitive_params, method='mode',
        figure_size=(4, 4), figure_path_prefixes=figure_paths,
        param_names_on_plot=calcium_models.params_on_plot,
        method_kwargs={'method': 'histogram'})
else:
    analyzer.run_sensitivity_test(
        calcium_ode, 0, ts, y0_run, y_run, 3, sensitive_params,
        figure_size=(4, 4), figure_path_prefixes=figure_paths,
        param_names_on_plot=calcium_models.params_on_plot)

# reset output directory
analyzer.output_dir = output_dir

# %%
# visualize stats from sensitivity test
sensitivity_dir = 'sensitivity-test'
sensitivity_dir = os.path.join(output_root, sensitivity_dir)
sensitivity_stats = {}
test_quantiles = [0.01, 0.99]

for param, qt in itertools.product(sensitive_params, test_quantiles):
    stats_path = os.path.join(sensitivity_dir, f'{param}_{qt}_stats.csv')
    sensitivity_stats[(param, qt)] = pd.read_csv(stats_path, index_col=0)

    # trajectory distances
    figure_path = os.path.join(sensitivity_dir, f'{param}_{qt}_traj_dists.pdf')
    plt.figure(figsize=(11, 8.5), dpi=300)
    plt.hist(sensitivity_stats[(param, qt)]['MeanTrajDist'])
    plt.savefig(figure_path)
    plt.close()

    # differences in peak
    figure_path = os.path.join(sensitivity_dir, f'{param}_{qt}_peak_diffs.pdf')
    plt.figure(figsize=(11, 8.5), dpi=300)
    plt.hist(sensitivity_stats[(param, qt)]['MeanPeakDiff'])
    plt.savefig(figure_path)
    plt.close()

    # fold changes in peak
    figure_path = os.path.join(sensitivity_dir,
                               f'{param}_{qt}_peak_fold_changes.pdf')
    plt.figure(figsize=(11, 8.5), dpi=300)
    plt.hist(sensitivity_stats[(param, qt)]['MeanPeakFoldChange'])
    plt.savefig(figure_path)
    plt.close()

    # differences in steady state
    figure_path = os.path.join(sensitivity_dir,
                               f'{param}_{qt}_steady_diffs.pdf')
    plt.figure(figsize=(11, 8.5), dpi=300)
    plt.hist(sensitivity_stats[(param, qt)]['MeanSteadyDiff'])
    plt.savefig(figure_path)
    plt.close()

    # differences in steady state
    figure_path = os.path.join(sensitivity_dir,
                               f'{param}_{qt}_steady_fold_changes.pdf')
    plt.figure(figsize=(11, 8.5), dpi=300)
    plt.hist(sensitivity_stats[(param, qt)]['MeanSteadyFoldChange'])
    plt.savefig(figure_path)
    plt.close()

# %%
# gather stats for trajectory distances from sensitivity test
test_quantiles = [0.01, 0.99]
sensitivity_dist_stats = pd.DataFrame(
    columns=['Mean', 'Std', 'Mode', 'Quantile_0.75'])

for param, qt in itertools.product(sensitive_params, test_quantiles):
    param_qt = f'{param}_{qt}'
    stats_path = os.path.join(sensitivity_dir, f'{param_qt}_stats.csv')
    stats = pd.read_csv(stats_path, index_col=0)
    mean_traj_dist = stats['MeanTrajDist']

    # calculate stats for the param-percentile combo
    sensitivity_dist_stats.loc[param_qt, 'Mean'] = mean_traj_dist.mean()
    sensitivity_dist_stats.loc[param_qt, 'Std'] = mean_traj_dist.std()
    sensitivity_dist_stats.loc[param_qt, 'Mode'] = \
        get_mode_continuous_rv(mean_traj_dist, method='histogram')
    sensitivity_dist_stats.loc[param_qt, 'Quantile_0.75'] = \
        mean_traj_dist.quantile(0.75)

    # make a histogram for trajectory distances
    figure_path = os.path.join(sensitivity_dir, f'{param_qt}_hist.pdf')
    plt.figure(figsize=(11, 8.5), dpi=300)
    plt.hist(mean_traj_dist, bins=20)
    plt.savefig(figure_path)
    plt.close()

sensitivity_dist_stats.to_csv(
    os.path.join(sensitivity_dir, 'sensitivity_dist_stats.csv'))

# %%
# make a heatmap for modes of trajectory distances
sensitivity_dist_stats = pd.read_csv(
    os.path.join(sensitivity_dir, 'sensitivity_dist_stats.csv'), index_col=0)
test_quantiles = [0.01, 0.99]
sensitivity_dist_modes = pd.DataFrame(index=sensitive_params,
                                      columns=test_quantiles, dtype=float)
sensitivity_dist_pct75 = pd.DataFrame(index=sensitive_params,
                                      columns=test_quantiles, dtype=float)
for param, qt in itertools.product(sensitive_params, test_quantiles):
    sensitivity_dist_modes.loc[param, qt] = \
        sensitivity_dist_stats.loc[f'{param}_{qt}', 'Mode']
    sensitivity_dist_pct75.loc[param, qt] = \
        sensitivity_dist_stats.loc[f'{param}_{qt}', 'Quantile_0.75']

figure_path = os.path.join(sensitivity_dir,
                           'sensitivity_dist_modes_heatmap.pdf')
plt.figure(figsize=(2.5, 7), dpi=300)
ax = plt.gca()
heatmap = ax.imshow(sensitivity_dist_modes, vmin=0)
ax.set_xticks(np.arange(2))
ax.set_xticklabels(test_quantiles)
ax.set_yticks(np.arange(len(sensitive_params)))
ax.set_yticklabels(
    [calcium_models.params_on_plot[p] for p in sensitive_params])
# draw text on each block
for i, j in itertools.product(range(len(sensitive_params)), range(2)):
    mode = sensitivity_dist_modes.iloc[i, j]
    pct75 = sensitivity_dist_pct75.iloc[i, j]
    ax.text(j, i, f'{mode:.2f}\n({pct75:.2f})', ha='center',
            va='center', color='w')
# plt.colorbar(heatmap, shrink=0.3)
plt.tight_layout()
plt.savefig(figure_path)
plt.close()

# %%
# simulate trajectories using posterior of previous cell
traj_pred_dir = os.path.join(output_root, 'prediction-by-predecessor')
if not os.path.exists(traj_pred_dir):
    os.mkdir(traj_pred_dir)

# simulate first cell by root cell
if first_cell_order == 1:
    y_sim = root_analyzer.simulate_chains(
        calcium_ode, 0, ts, np.array([0, 0, 0.7, y0_all[mixed_cells[0]]]),
        subsample_step_size=50, plot=False, verbose=False)
    for chain in range(analyzer.num_chains):
        figure_path = os.path.join(
            traj_pred_dir, f'0001_cell_{mixed_cells[0]}_{chain}.pdf')
        fig = Figure(figsize=(3, 2), dpi=300)
        ax = fig.gca()
        ax.plot(ts, y_all[mixed_cells[0], :], 'ko', fillstyle='none')
        ax.plot(ts, y_sim[chain][:, :, 3].T, color='C0', alpha=0.05)
        fig.savefig(figure_path)
        plt.close(fig)

# simulate all other cells
figure_paths = []
next_cells = mixed_cells[1:]
for i, cell in enumerate(next_cells):
    cell_fps = []
    cell_order = first_cell_order + i + 1
    for chain in range(analyzer.num_chains):
        chain_fp = os.path.join(
            traj_pred_dir, f'{cell_order:04d}_cell_{cell}_{chain}.pdf')
        cell_fps.append(chain_fp)
    figure_paths.append(cell_fps)

y0_run = np.zeros((num_mixed_cells - 1, 4))
y0_run[:, 2] = 0.7
y0_run[:, 3] = y0_all[next_cells]
y_run = y_all[next_cells, :]
analyzer.simulate_trajectories(calcium_ode, 0, ts, y0_run, y_run, 3,
                               exclude_sessions=[str(mixed_cells[-1])],
                               output_paths=figure_paths)

# %%
# plot trajectory from sample mean for sampling vs prediction
for i, cell in enumerate(mixed_cells):
    # simulate trajectory using sample mean
    y_sample_mean = \
        analyzer.session_analyzers[i].get_sample_mean_trajectory(
            calcium_ode, 0, ts, np.array([0, 0, 0.7, y0_all[cell]]))

    # simulate trajectory using sample mean of previous cell
    if i == 0:
        y_sample_mean_pred = root_analyzer.get_sample_mean_trajectory(
                calcium_ode, 0, ts, np.array([0, 0, 0.7, y0_all[cell]]))
    else:
        y_sample_mean_pred = \
            analyzer.session_analyzers[i - 1].get_sample_mean_trajectory(
                calcium_ode, 0, ts, np.array([0, 0, 0.7, y0_all[cell]]))

    # plot the trajectories
    figure_path = os.path.join(output_root, 'prediction-by-predecessor',
                               f'{i:04d}_cell_{cell:04d}_vs_sampled.pdf')
    plt.figure(figsize=(11, 8.5), dpi=300)
    plt.plot(ts, y_all[cell, :], label='True')
    plt.plot(ts, y_sample_mean[:, 3], label='Sampled')
    plt.plot(ts, y_sample_mean_pred[:, 3], label='Predicted')
    plt.legend()
    plt.savefig(figure_path)
    plt.close()

# %%
# get trajectory distances if predicted by sample from predecessors
traj_pred_dists = np.empty(num_mixed_cells)

# root cell
traj_pred_dists[0] = root_analyzer.get_trajectory_distance(
    calcium_ode, 0, ts, np.array([0, 0, 0.7, y0_all[mixed_cells[0]]]),
    y_all[mixed_cells[0], :], 3)

# all other cells
for i, cell in enumerate(mixed_cells[1:]):
    traj_pred_dists[i + 1] = \
        analyzer.session_analyzers[i].get_trajectory_distance(
            calcium_ode, 0, ts, np.array([0, 0, 0.7, y0_all[cell]]),
            y_all[cell, :], 3)

traj_dist_path = os.path.join(output_dir, 'mean_trajectory_distances.csv')
traj_stat_tables = pd.read_csv(traj_dist_path, index_col=0, header=None,
                         squeeze=True)

# %%
# histogram of distance from predicted to true
figure_path = os.path.join(
    output_dir, 'mean_predicted_trajectory_distance.pdf')
plt.figure(figsize=(11, 8.5), dpi=300)
plt.hist(traj_pred_dists, bins=25, range=(-50, 50))
plt.savefig(figure_path)
plt.close()

# violin plot for sampled vs predicted
figure_path = os.path.join(
    output_dir, 'mean_trajectory_distances_sampled_vs_prediction.pdf')
plt.figure(figsize=(11, 8.5), dpi=300)
plt.violinplot([traj_stat_tables, traj_pred_dists])
plt.xticks(ticks=[1, 2], labels=['From posterior', 'From prior'])
plt.savefig(figure_path)
plt.close()

# histogram of sampled vs predicted
traj_dist_diffs = traj_pred_dists - traj_stat_tables
figure_path = os.path.join(
    output_dir, 'mean_trajectory_distances_sampled_vs_prediction_diff.pdf')
plt.figure(figsize=(11, 8.5), dpi=300)
plt.hist(traj_dist_diffs, bins=25, range=(-50, 50))
plt.savefig(figure_path)
plt.close()

# %%
def predict_unsampled(num_unsampled, num_sampled, figure_dir,
                      use_similar_cell=True, sample_marginal=False,
                      random_seed=0, plot_traj=False):
    bit_generator = np.random.MT19937(random_seed)
    rng = np.random.default_rng(bit_generator)
    unsampled_cells = full_cell_list.loc[last_cell_order + 1:, 'Cell']
    unsampled_cells = unsampled_cells.sample(n=num_unsampled,
                                             random_state=bit_generator)
    sampled_cell_list = cell_list.loc[:num_sampled, 'Cell'].to_numpy()
    traj_stats_table = pd.DataFrame(
        index=range(num_unsampled),
        columns=['Cell', 'SampledCell', 'Distance', 'PeakDiff',
                 'PeakFoldChange', 'PeakTimeDiff', 'SteadyDiff',
                 'SteadyFoldChange', 'DerivativeDist'])

    for cell_idx, cell in enumerate(tqdm(unsampled_cells)):
        if use_similar_cell:
            # find the most similar cell among sampled ones
            cells_by_similarity = np.argsort(similarity_matrix[cell, :])[::-1]
            sampled_cell = next(c for c in cells_by_similarity
                                if c != cell and c in sampled_cell_list)
        else:
            # find a random sampled cell
            sampled_cell = rng.choice(sampled_cell_list)

        # predict using the selected cell
        sampled_cell_order = cell_list['Cell'].to_list().index(sampled_cell)
        sampled_cell_analyzer = analyzer.session_analyzers[sampled_cell_order]
        mixed_chains = sampled_cell_analyzer.get_mixed_chains()
        y0_cell = np.array([0, 0, 0.7, y0_all[cell]])
        y_cell = y_all[cell, :]

        if sample_marginal:
            # predict by samples from marginal distribution
            subsample_size = 10
            traj_pred = []

            for chain_idx in mixed_chains:
                sample = sampled_cell_analyzer.samples[chain_idx]
                subsample = np.empty((subsample_size, num_params - 1))
                traj_chain = np.empty((subsample_size, ts.size))

                # get subsample from marginal distribution for each parameter
                for param_idx, param in enumerate(param_names[1:]):
                    subsample[:, param_idx] = sample[param].sample(
                        subsample_size, random_state=bit_generator)

                # simulate from subsample
                for theta_idx, theta in enumerate(subsample):
                    traj_sample = simulate_trajectory(calcium_ode, theta, 0,
                                                      ts, y0_cell)
                    traj_chain[theta_idx, :] = traj_sample[:, 3]

                traj_pred.append(traj_chain)

            traj_pred = np.concatenate(traj_pred)
        else:
            # predict by actual draws from sample
            traj_pred = sampled_cell_analyzer.simulate_chains(
                calcium_ode, 0, ts, y0_cell, subsample_step_size=50,
                plot=False, verbose=False)
            traj_pred = np.concatenate(
                [traj_pred[c][:, : ,3] for c in mixed_chains])

        # plot predicted trajectories
        if plot_traj:
            figure_name = f'{cell:04d}_{num_sampled}_{sampled_cell:04d}'
            if sample_marginal:
                figure_name += '_marginal'
            figure_name += '.pdf'
            figure_path = os.path.join(figure_dir, figure_name)
            plt.figure()
            plt.plot(ts, traj_pred.T, color='C0')
            plt.plot(ts, y_cell, 'ko', fillstyle='none')
            plt.savefig(figure_path)
            plt.close()

        # get stats of predicted trajectories
        traj_stats_table.loc[cell_idx, 'Cell'] = cell
        traj_stats_table.loc[cell_idx, 'SampledCell'] = sampled_cell

        # trajectory distances
        traj_dist = np.mean(np.linalg.norm(traj_pred - y_cell, axis=1))
        traj_stats_table.loc[cell_idx, 'Distance'] = traj_dist

        # peaks
        traj_ref_peak = np.amax(y_cell)
        traj_pred_peaks = np.amax(traj_pred, axis=1)
        traj_peak_diffs = traj_pred_peaks - traj_ref_peak
        traj_stats_table.loc[cell_idx, 'PeakDiff'] = np.mean(traj_peak_diffs)
        traj_peak_fold_changes = np.log2(traj_pred_peaks / traj_ref_peak)
        traj_stats_table.loc[cell_idx, 'PeakFoldChange'] = \
            np.mean(traj_peak_fold_changes)
        traj_ref_peak_time = np.amax(y_cell[:100])
        traj_pred_peak_times = np.amax(traj_pred[:, :100], axis=1)
        traj_stats_table.loc[cell_idx, 'PeakTimeDiff'] = \
            np.mean(traj_pred_peak_times - traj_ref_peak_time)

        # steady states
        num_steady_pts = ts.size // 5
        traj_ref_steady_state = np.mean(y_cell[-num_steady_pts:])
        traj_pred_steady_states = np.mean(traj_pred[:, -num_steady_pts:],
                                          axis=1)
        traj_steady_diffs = traj_pred_steady_states - traj_ref_steady_state
        traj_stats_table.loc[cell_idx, 'SteadyDiff'] = \
            np.mean(traj_steady_diffs)
        traj_steady_fold_changes = \
            np.log2(traj_pred_steady_states / traj_ref_steady_state)
        traj_stats_table.loc[cell_idx, 'SteadyFoldChange'] = \
            np.mean(traj_steady_fold_changes)

        # L^2 distances between derivatives
        traj_prime = np.gradient(traj_pred[:, :100], axis=1)
        traj_prime_dist = np.mean(
            np.linalg.norm(traj_prime - y_prime[cell, :100], axis=1))
        traj_stats_table.loc[cell_idx, 'DerivativeDist'] = traj_prime_dist

    return traj_stats_table

# %%
# predict unsampled cells using similar or random cells
rng_seed = 0
num_unsampled = 500
unsampled_pred_dir = os.path.join(
    output_root, f'prediction-unsampled-{num_unsampled}-{rng_seed}')
if not os.path.exists(unsampled_pred_dir):
    os.mkdir(unsampled_pred_dir)

unsampled_traj_stats = {}
unsampled_traj_stats['similar'] = {}
unsampled_traj_stats['random'] = {}
unsampled_traj_stats['marginal'] = {}
num_sampled_for_prediction = [500, 400, 300, 200, 100]

for n in num_sampled_for_prediction:
    unsampled_traj_stats['similar'][n] = predict_unsampled(
        num_unsampled, n, unsampled_pred_dir, random_seed=rng_seed)
    table_path = os.path.join(unsampled_pred_dir,
                              f'trajectory_stats_similar_{n}.csv')
    unsampled_traj_stats['similar'][n].to_csv(table_path)

    unsampled_traj_stats['random'][n] = predict_unsampled(
        num_unsampled, n, unsampled_pred_dir, use_similar_cell=False,
        random_seed=rng_seed)
    table_path = os.path.join(unsampled_pred_dir,
                              f'trajectory_stats_random_{n}.csv')
    unsampled_traj_stats['random'][n].to_csv(table_path)

    unsampled_traj_stats['marginal'][n] = predict_unsampled(
        num_unsampled, n, unsampled_pred_dir, use_similar_cell=False,
        sample_marginal=True, random_seed=rng_seed)
    table_path = os.path.join(unsampled_pred_dir,
                              f'trajectory_stats_marginal_{n}.csv')
    unsampled_traj_stats['marginal'][n].to_csv(table_path)

# %%
# predict using lemon prior
def predict_unsampled_lemon(num_unsampled, random_seed=0):
    bit_generator = np.random.MT19937(random_seed)
    unsampled_cells = full_cell_list.loc[last_cell_order + 1:, 'Cell']
    unsampled_cells = unsampled_cells.sample(n=num_unsampled,
                                             random_state=bit_generator)
    traj_stats_table = pd.DataFrame(
        index=range(num_unsampled),
        columns=['Cell', 'SampledCell', 'Distance', 'PeakDiff',
                 'PeakFoldChange', 'SteadyDiff', 'SteadyFoldChange',
                 'DerivativeDist'])

    lemon_prior_spec_path = os.path.join('stan_models', ode_variant,
                                         'calcium_model_alt_prior.txt')
    lemon_prior_spec = pd.read_csv(lemon_prior_spec_path, delimiter='\t',
                                index_col=0)
    lemon_prior = lemon_prior_spec['mu'].to_numpy()

    for cell_idx, cell in enumerate(tqdm(unsampled_cells)):
        traj_pred = simulate_trajectory(calcium_ode, lemon_prior, 0, ts,
                                        np.array([0, 0, 0.7, y0_all[cell]]))
        traj_pred = traj_pred[:, 3]
        y_cell = y_all[cell, :]

        # get stats of predicted trajectories
        traj_stats_table.loc[cell_idx, 'Cell'] = cell
        traj_stats_table.loc[cell_idx, 'SampledCell'] = 'Lemon'
        traj_dist = np.linalg.norm(traj_pred - y_cell)
        traj_stats_table.loc[cell_idx, 'Distance'] = traj_dist
        # peaks
        traj_ref_peak = np.amax(y_cell)
        traj_pred_peaks = np.amax(traj_pred)
        traj_peak_diffs = traj_pred_peaks - traj_ref_peak
        traj_stats_table.loc[cell_idx, 'PeakDiff'] = traj_peak_diffs
        traj_peak_fold_changes = traj_pred_peaks / traj_ref_peak
        if traj_peak_fold_changes < 1:
            traj_peak_fold_changes = -1 / traj_peak_fold_changes
        traj_stats_table.loc[cell_idx, 'PeakFoldChange'] = \
            traj_peak_fold_changes
        # steady states
        num_steady_pts = ts.size // 5
        traj_ref_steady_state = np.mean(y_cell[-num_steady_pts:])
        traj_pred_steady_states = np.mean(traj_pred[-num_steady_pts:])
        traj_steady_diffs = traj_pred_steady_states - traj_ref_steady_state
        traj_stats_table.loc[cell_idx, 'SteadyDiff'] = traj_steady_diffs
        traj_steady_fold_changes = \
            traj_pred_steady_states / traj_ref_steady_state
        if traj_steady_fold_changes < 1:
            traj_steady_fold_changes = -1 / traj_steady_fold_changes
        traj_stats_table.loc[cell_idx, 'SteadyFoldChange'] = \
            traj_steady_fold_changes

    return traj_stats_table

# %%
unsampled_traj_stats['lemon'] = predict_unsampled_lemon(
    num_unsampled, random_seed=rng_seed)
table_path = os.path.join(unsampled_pred_dir, f'trajectory_stats_lemon.csv')
unsampled_traj_stats['lemon'].to_csv(table_path)

# %%
# load saved statistics if necessary
rng_seed = 0
num_unsampled = 500
unsampled_pred_dir = os.path.join(
    output_root, f'prediction-unsampled-{num_unsampled}-{rng_seed}')
unsampled_traj_stats = {}
unsampled_traj_stats['similar'] = {}
unsampled_traj_stats['random'] = {}
unsampled_traj_stats['marginal'] = {}
num_sampled_for_prediction = [500, 400, 300, 200, 100]

# load distances from 'similar', 'random', 'marginal'
for n in num_sampled_for_prediction:
    for method in unsampled_traj_stats:
        table_path = os.path.join(unsampled_pred_dir,
                                  f'trajectory_stats_{method}_{n}.csv')
        unsampled_traj_stats[method][n] = pd.read_csv(table_path, index_col=0)

# load distances from 'lemon'
table_path = os.path.join(unsampled_pred_dir, f'trajectory_stats_lemon.csv')
unsampled_traj_stats['lemon'] = pd.read_csv(table_path)

# %%
# plot stats for predictions
prediction_methods = ['similar', 'random', 'marginal', 'lemon']
stats_names = ['Distance', 'PeakDiff', 'PeakFoldChange', 'PeakTimeDiff',
               'SteadyDiff', 'SteadyFoldChange', 'DerivativeDist']
stats_fig_names = ['mean_trajectory_distances', 'mean_peak_diffs',
                   'mean_peak_fold_changes', 'peak_time_diff',
                   'mean_steady_diffs', 'mean_steady_fold_changes',
                   'mean_derivative_distances']
stats_xlabels = ['Mean trajectory distance', 'Mean peak difference',
                 'Mean peak fold change', 'Man peak time difference',
                 'Mean steady state difference',
                 'Mean steady state fold change',
                 'Mean first derivative distance']

prediction_stats_summary = {}
for method in prediction_methods[:-1]:
    prediction_stats_summary[method] = {}
    for n in num_sampled_for_prediction:
        prediction_stats_summary[method][n] = pd.DataFrame(
            columns=['Mean', 'StdDev', 'Median'], index=stats_names)

for method, traj_stat_tables in unsampled_traj_stats.items():
    if method == 'lemon':
        # make histogram for prediction stats
        for i in range(len(stats_names)):
            if stats_names[i] not in ['PeakTimeDiff', 'DerivativeDist']:
                figure_path = os.path.join(
                    unsampled_pred_dir,
                    f'{stats_fig_names[i]}_prediction_unsampled_lemon.pdf')
                plt.figure(figsize=(6, 4), dpi=300)
                plt.hist(traj_stat_tables[stats_names[i]], bins=20)
                plt.xlabel(stats_xlabels[i])
                plt.ylabel('Number of cells')
                plt.tight_layout()
                plt.savefig(figure_path)
                plt.close()
    else:
        for n in num_sampled_for_prediction:
            for i, stat in enumerate(stats_names):
                # get summary for prediction stats
                prediction_stats_summary[method][n].loc[stat, 'Mean'] = \
                    traj_stat_tables[n][stat].mean()
                prediction_stats_summary[method][n].loc[stat, 'StdDev'] = \
                    traj_stat_tables[n][stat].std()
                prediction_stats_summary[method][n].loc[stat, 'Median'] = \
                    traj_stat_tables[n][stat].std()

                # make histogram for prediction stats
                figure_path = os.path.join(
                    unsampled_pred_dir,
                    f'{stats_fig_names[i]}_prediction_unsampled_{method}'
                        + f'_{n}.pdf')
                plt.figure(figsize=(6, 4), dpi=300)
                plt.hist(traj_stat_tables[n][stat], bins=20)
                plt.xlabel(stats_xlabels[i])
                plt.ylabel('Number of cells')
                plt.tight_layout()
                plt.savefig(figure_path)
                plt.close()

            # export summary for prediction stats
            summary_path = os.path.join(
                unsampled_pred_dir, f'stats_summary_{method}_{n}.csv')
            prediction_stats_summary[method][n].to_csv(summary_path)

            # make histogram of similarity
            figure_path = os.path.join(
                unsampled_pred_dir, f'unsampled_{method}_similarity_{n}.pdf')
            plt.figure(figsize=(6, 4), dpi=300)
            similarity = [similarity_matrix[row.Cell, row.SampledCell]
                          for row in traj_stat_tables[n].itertuples()]
            plt.hist(similarity)
            plt.tight_layout()
            plt.savefig(figure_path)
            plt.close()

# make histogram for multiple stats in the same figure
for n in num_sampled_for_prediction:
    for i in range(len(stats_names)):
        # plot all methods together
        if stats_names[i] not in ['PeakTimeDiff', 'DerivativeDist']:
            figure_path = os.path.join(
                unsampled_pred_dir,
                f'{stats_fig_names[i]}_prediction_unsampled_{n}.pdf')
            plt.figure(figsize=(6, 4), dpi=300)
            # gather data
            traj_plot_stats = [unsampled_traj_stats[method][n][stats_names[i]]
                            for method in prediction_methods[:-1]]
            traj_plot_stats = np.vstack(traj_plot_stats)
            traj_plot_stats = np.vstack([
                traj_plot_stats, unsampled_traj_stats['lemon'][stats_names[i]]])
            traj_plot_stats = traj_plot_stats.T
            plt.hist(traj_plot_stats, label=prediction_methods, bins=20)
            # for method, traj_stat_tables in unsampled_traj_stats.items():
            #     if method == 'similar':
            #         zorder = 3
            #     elif method == 'random':
            #         zorder = 2
            #     else:
            #         zorder = 1

            #     if method == 'lemon':
            #         plt.hist(traj_stat_tables[stats_names[i]], bins=20, alpha=0.5,
            #                  label=method.capitalize())
            #     else:
            #         plt.hist(traj_stat_tables[n][stats_names[i]],bins=20,
            #                  alpha=0.5, label=method.capitalize(),
            #                  zorder=zorder)
            plt.legend()
            plt.xlabel(stats_xlabels[i])
            plt.ylabel('Number of cells')
            plt.tight_layout()
            plt.savefig(figure_path)
            plt.close()

        # plot similar vs random
        figure_path = os.path.join(
            unsampled_pred_dir,
            f'{stats_fig_names[i]}_prediction_unsampled_similar_vs_' +
            f'random_{n}.pdf')
        plt.figure(figsize=(6, 4), dpi=300)
        traj_plot_stats = [unsampled_traj_stats[method][n][stats_names[i]]
                           for method in prediction_methods[:2]]
        traj_plot_stats = np.vstack(traj_plot_stats)
        traj_plot_stats = traj_plot_stats.T
        plt.hist(traj_plot_stats, label=prediction_methods[:2], bins=20)
        # for method in ['similar', 'random']:
        #     traj_stat_tables = unsampled_traj_stats[method]
        #     if method == 'similar':
        #         zorder = 3
        #     else:
        #         zorder = 2

        #     plt.hist(traj_stat_tables[n][stats_names[i]], bins=20, alpha=0.5,
        #              label=method.capitalize(), zorder=zorder)
        plt.legend()
        plt.xlabel(stats_xlabels[i])
        plt.ylabel('Number of cells')
        plt.tight_layout()
        plt.savefig(figure_path)
        plt.close()

        # plot similar vs marginal and lemon
        if stats_names[i] not in ['PeakTimeDiff', 'DerivativeDist']:
            figure_path = os.path.join(
                unsampled_pred_dir,
                f'{stats_fig_names[i]}_prediction_unsampled_similar_vs_' +
                f'others_{n}.pdf')
            plt.figure(figsize=(6, 4), dpi=300)
            traj_plot_stats = [unsampled_traj_stats[method][n][stats_names[i]]
                               for method in ['similar', 'marginal']]
            traj_plot_stats = np.vstack(traj_plot_stats)
            traj_plot_stats = np.vstack([
                traj_plot_stats, unsampled_traj_stats['lemon'][stats_names[i]]])
            traj_plot_stats = traj_plot_stats.T
            plt.hist(traj_plot_stats, label=['similar', 'marginal', 'lemon'],
                    bins=20)
            # for method in ['similar', 'marginal', 'lemon']:
            #     traj_stat_tables = unsampled_traj_stats[method]

            #     if method == 'similar':
            #         zorder = 3
            #     else:
            #         zorder = 2

            #     if method == 'lemon':
            #         plt.hist(traj_stat_tables[stats_names[i]], bins=20, alpha=0.5,
            #                  label=method.capitalize())
            #     else:
            #         plt.hist(traj_stat_tables[n][stats_names[i]], bins=20,
            #                  alpha=0.5, label=method.capitalize(),
            #                  zorder=zorder)
            plt.legend()
            plt.xlabel(stats_xlabels[i])
            plt.ylabel('Number of cells')
            plt.tight_layout()
            plt.savefig(figure_path)
            plt.close()

    # make scatter plots for similarity vs trajectory distances
    # figure_path = os.path.join(unsampled_pred_dir,
    #                         f'similarity_vs_traj_dists_{n}.pdf')
    # plt.figure(figsize=(6, 4))
    # for method, traj_dist_tables in unsampled_traj_dists.items():
    #     if method == 'lemon':
    #         continue

    #     similarity = [similarity_matrix[row.Cell, row.SampledCell]
    #                 for row in traj_dist_tables[n].itertuples()]
    #     plt.scatter(similarity, traj_dist_tables[n]['Distance'],
    #                 alpha=0.5, label=method.capitalize())
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(figure_path)
    # plt.close()

# %%
# get K-S stats
from scipy.stats import ks_2samp
stats_names = ['Distance', 'PeakDiff', 'PeakFoldChange', 'PeakTimeDiff',
               'SteadyDiff', 'SteadyFoldChange', 'DerivativeDist']
unsampled_stats_ks_stats = {}
ks_alt_hypothesis = 'greater'

for n in num_sampled_for_prediction:
    unsampled_stats_ks_stats[n] = pd.DataFrame(columns=['K-S', 'p-value'],
                                                    index=stats_names)
    for stat in stats_names:
        ks, p = ks_2samp(unsampled_traj_stats['similar'][n][stat],
                         unsampled_traj_stats['random'][n][stat],
                         alternative=ks_alt_hypothesis)
        unsampled_stats_ks_stats[n].loc[stat, 'K-S'] = ks
        unsampled_stats_ks_stats[n].loc[stat, 'p-value'] = p

    unsampled_stats_ks_stats_path = os.path.join(
        unsampled_pred_dir,
        f'trajectory_stats_similar_vs_random_{n}_ks_{ks_alt_hypothesis}.csv')
    unsampled_stats_ks_stats[n].to_csv(unsampled_stats_ks_stats_path)

# %%
# use multiple cells to predict one cell
rng_seed = 0
num_unsampled = 50
unsampled_pred_dir = os.path.join(
    output_root, f'prediction-unsampled-batch-{num_unsampled}-{rng_seed}')
if not os.path.exists(unsampled_pred_dir):
    os.mkdir(unsampled_pred_dir)

def predict_unsampled_batch(num_sampled, num_predictions, random_seed=0):
    bit_generator = np.random.MT19937(random_seed)
    unsampled_cells = full_cell_list.loc[last_cell_order + 1:, 'Cell']
    unsampled_cells = unsampled_cells.sample(n=num_unsampled,
                                             random_state=bit_generator)

    traj_dist_table = pd.DataFrame(columns=['Cell', 'SampledCell', 'Distance'])
    traj_dist_table.astype({'Cell': int, 'SampledCell': int})

    sampled_cell_list = cell_list.loc[:num_sampled, 'Cell'].sample(
        num_predictions, random_state=bit_generator)
    for cell in unsampled_cells:
        for sampled_cell in tqdm(sampled_cell_list):
            sampled_cell_order = cell_list['Cell'].to_list().index(sampled_cell)
            sampled_cell_analyzer = analyzer.session_analyzers[
                sampled_cell_order]

            # plot predicted trajectories
            traj_pred = sampled_cell_analyzer.simulate_chains(
                calcium_ode, 0, ts, np.array([0, 0, 0.7, y0_all[cell]]),
                subsample_step_size=50, plot=False, verbose=False)
            mixed_chains = sampled_cell_analyzer.get_mixed_chains()
            traj_pred_mixed = np.concatenate(
                [traj_pred[c][:, : ,3] for c in mixed_chains])

            traj_pred_path = os.path.join(
                unsampled_pred_dir,
                f'{cell:04d}_{num_sampled}_{sampled_cell:04d}.pdf')
            plt.figure()
            plt.plot(ts, traj_pred_mixed.T, color='C0')
            plt.plot(ts, y_all[cell, :], 'ko', fillstyle='none')
            plt.savefig(traj_pred_path)
            plt.close()

            # get trajectory distances
            traj_dist = sampled_cell_analyzer.get_trajectory_distance(
                calcium_ode, 0, ts, np.array([0, 0, 0.7, y0_all[cell]]),
                y_all[cell, :], 3)
            traj_dist_row = {'Cell': cell, 'SampledCell': sampled_cell,
                             'Distance': traj_dist}
            traj_dist_table = traj_dist_table.append(traj_dist_row,
                                                     ignore_index=True)

    return traj_dist_table

traj_dist_unsampled_batch = predict_unsampled_batch(last_cell_order, 100)

# %%
# make histogram of trajectory distances for predicted cells
for cell in traj_dist_unsampled_batch['Cell'].unique():
    traj_stat_tables = traj_dist_unsampled_batch.loc[
        traj_dist_unsampled_batch['Cell'] == cell, 'Distance']

    figure_path = os.path.join(unsampled_pred_dir,
                               f'{cell:04d}_trajectory_distances.pdf')
    plt.figure(figsize=(11, 8.5), dpi=300)
    plt.hist(traj_stat_tables, range=(0, 30))
    plt.savefig(figure_path)
    plt.close()

# %%
# make plots for basic stats
print('Plotting sampling time...')
analyzer.plot_sampling_time(time_unit='m', xticks=xticks, hist_range=(0, 300))
print('Plotting mean tree depths...')
analyzer.plot_mean_tree_depths(tree_depth_min=0, tree_depth_max=15,
                               xticks=xticks)
print('Plotting mean log posteriors...')
analyzer.plot_mean_log_posteriors(xticks=xticks)
print('Plotting R^hat of posterior')
analyzer.plot_posterior_rhats(xticks=xticks)

print('Plotting mean distances between true and simulated trajectories...')
analyzer.plot_mean_trajectory_distances(
    calcium_ode, 0, ts, y0_all[mixed_cells], y_all[mixed_cells, :], dist_min=0,
    dist_max=50, xticks=xticks)
print('Plotting R^hats vs mean trajectory distances...')
lp_rhats_vs_traj_dists_stats = analyzer.plot_lp_rhats_vs_trajectory_distances(
        calcium_ode, 0, ts, y0_all[mixed_cells], y_all[mixed_cells, :])
print('Plotting mean log posteriors vs mean trajectory distances...')
mean_lps_vs_traj_dists_stats = analyzer.plot_mean_lps_vs_trajectory_distances(
        calcium_ode, 0, ts, y0_all[mixed_cells], y_all[mixed_cells, :])
print('Plotting mean log posteriors vs R^hats...')
mean_lps_vs_lp_rhats = analyzer.plot_mean_lps_vs_lp_rhats()

# %%
param_plot_titles = [stan_run_meta[stan_run]['pub_name']] * num_params
# print('Making violin plot of sampled parameters...')
# analyzer.plot_parameter_violin(page_size=(6, 2), num_rows=1, num_cols=1,
#                                xticks=xticks, titles=param_plot_titles,
#                                y_labels=param_names)
# print('Making ribbon plot of sampled parameters...')
# analyzer.plot_parameter_ribbon(page_size=(6, 2), num_rows=1, num_cols=1)
print('Making box plot of sampled parameters...')
analyzer.plot_parameter_box(
    page_size=(6, 2), num_rows=1, num_cols=1, xticks=xticks,
    titles=param_plot_titles,
    y_labels=[calcium_models.params_on_plot[p] for p in param_names])

# %%
print('Plotting select pairs of parameters...')
session_param_pairs_dir = os.path.join(output_root, 'param-pairs')
if not os.path.exists(session_param_pairs_dir):
    os.mkdir(session_param_pairs_dir)

output_path_prefixes = [
    os.path.join(session_param_pairs_dir, f'{idx:04d}_cell_{session}')
    for idx, session in enumerate(analyzer.session_list)]
analyzer.plot_param_pairs_all_sessions(
    select_param_pairs, output_path_prefixes=output_path_prefixes,
    param_names_on_plot=calcium_models.params_on_plot)

# %%
analyzer.get_sample_means()

# %%
print('Plotting select pairs of parameters...')
# param_pair_sessions = analyzer.session_list[::50].tolist()
param_pair_sessions =['5121', '5104', '4996', '4962', '4918', '4824', '4800',
                      '4881', '4531', '4571']
param_plot_titles = ['Sample means'] \
    + [f'Cell {c}' for c in param_pair_sessions]
for pairs in select_param_pairs:
    analyzer.plot_param_pairs(
        pairs, sessions=param_pair_sessions, num_rows=1, num_cols=1,
        page_size=(4, 4), param_names_on_plot=calcium_models.params_on_plot,
        titles=param_plot_titles)

# %%
# make legend gene-param plot with Huber regression
plt.figure(figsize=(2, 0.5), dpi=300)
gradient = np.linspace(0, 1, 100)
gradient = gradient[np.newaxis, :]
plt.imshow(gradient, aspect=3.0, cmap=plt.get_cmap('viridis'))
plt.axis('off')
plt.title('Cell positions', fontdict={'fontsize': 'medium'})
figure_path = os.path.join(
    output_dir, 'param_pair_scatters_legend.pdf')
plt.tight_layout()
plt.savefig(figure_path)
plt.close()

# %%

# %%
print('Loading gene expression data and preprocessing...')
analyzer.load_expression_data('../../data/vol_adjusted_genes_transpose.txt')
print('Filtering sessions with extreme samples...')
analyzer.filter_sessions(z_score_max=3.0)
print('Plotting correlation between sampled parameters...')
analyzer.get_parameter_correlations()

print('Getting top genes from PCA...')
analyzer.run_pca(sampled_only=pca_sampled_only)
analyzer.get_top_genes_from_pca()
print('Computing correlation between top genes and parameters...')
analyzer.compute_gene_param_correlations(analyzer.top_pc_gene_list)

print('Running regression for genes vs parameters...')
# high_corr_pairs = [('CAPN1', 'd5'), ('PPP1CC', 'L'), ('ITPRIPL2', 'k3'),
#                    ('MSH2', 'd5'), ('PRKCI', 'd5'), ('PPP2CA', 'd5'),
#                    ('PPP2CA', 'k3'), ('PRKCI', 'a'), ('PPP1CC', 'eta1'),
#                    ('PPP1CC', 'd5'), ('ATP2B1', 'd5'), ('CCDC47', 'd5'),
#                    ('RCN1', 'd5'), ('PPP1CC', 'a'), ('PPP2CB', 'd5'),
#                    ('PPP3CA', 'd5'), ('PPP1CC', 'k3'), ('ATP2C1', 'd5')]
# high_corr_pairs.sort()
# high_corr_pairs = [p for p in high_corr_pairs if p[1] in param_names]
num_top_pairs = 20
high_corr_pairs = []
for i in range(num_top_pairs):
    gene = analyzer.sorted_gene_vs_param_pairs.loc[i, 'Gene']
    param = analyzer.sorted_gene_vs_param_pairs.loc[i, 'Parameter']
    high_corr_pairs.append((gene, param))

regressors_trained = analyzer.run_genes_vs_params_regression(
    'huber', analyzer.top_pc_gene_list, select_pairs=high_corr_pairs)
print('Plotting select pairs of genes and parameters...')
scatter_kwargs = {'s': 3.0}
analyzer.plot_select_genes_vs_params(
    high_corr_pairs, regressors_trained, 'high_corr_pairs_scatter_huber.pdf',
    figure_size=(2.5, 2.5), num_rows=1, num_cols=1, show_corrs=False,
    param_names_on_plot=calcium_models.params_on_plot, **scatter_kwargs)

print('All done!')

# %%
# make legend gene-param plot with Huber regression
import matplotlib.patches as mpatches

plt.figure(figsize=(2.5, 1), dpi=300)
gradient = np.linspace(0, 1, 100)
gradient = gradient[np.newaxis, :]
plt.imshow(gradient, aspect=3.0, cmap=plt.get_cmap('viridis'))
plt.axis('off')
plt.title('Cell positions', fontdict={'fontsize': 'medium'})
legend_patches = [mpatches.Patch(color='C1', label='Huber')]
plt.legend(legend_patches, ['Huber Regression'], loc='upper left',
           frameon=False, bbox_to_anchor=(0.0, 0.0))
figure_path = os.path.join(
    output_dir, 'high_corr_pairs_scatter_huber_legend.pdf')
plt.tight_layout()
plt.savefig(figure_path)
plt.close()

# %%
# generate LaTeX code for table of top 20 gene-param pairs
for i, row in analyzer.sorted_gene_vs_param_pairs.iterrows():
    gene = row['Gene']
    param = row['Parameter']
    param = calcium_models.params_on_plot[param].replace('mathrm', 'text')
    corr = row['Correlation']
    p_val = row['p-value']
    line = f"        {gene} & {param} & ${corr:.6f}$ & ${p_val:.6e}}}$ \\\\"
    line = line.replace('e-', ' \\times 10^{-')
    line = line.replace('{-0', '{-')
    print(line)

    if i == 19:
        break

# %%
# analyze warmup
warmup_time = pd.DataFrame(index=analyzer.session_list,
                           columns=range(analyzer.num_chains))
warmup_iters = 500

for idx, a in zip(analyzer.session_list, analyzer.session_analyzers):
    # compute mean and standard deviation of log posteriors
    lps = a.get_log_posteriors(include_warmup=True)
    mixed_chains = a.get_mixed_chains()

    # find first iteration such that the log posterior is within 3 standard
    # deviations from mean
    for chain in range(analyzer.num_chains):
        if chain in mixed_chains:
            lp_mean = np.mean(lps[chain, warmup_iters:])
            lp_std = np.std(lps[chain, warmup_iters:])
            lp_z_scores = np.abs((lps[chain, :] - lp_mean) / lp_std)
            warmup_time.loc[idx, chain] = np.argwhere(lp_z_scores < 3)[0][0]

output_path = os.path.join(output_dir, 'warmup_time.csv')
warmup_time.to_csv(output_path)

# %%
# plot warmup time
plt.figure(figsize=(6, 4), dpi=300)
if warmup_time.shape[0] > 100:
    warmup_time_sample = warmup_time.sample(n=100)
else:
    warmup_time_sample = warmup_time
plt.hist(warmup_time_sample.to_numpy().flatten(), bins=50,
         range=(0, warmup_iters))
plt.ylim((0, 200))
plt.xlabel('Warmup time')
plt.ylabel('Number of chains')
plt.tight_layout()
figure_path = os.path.join(analyzer.output_dir, 'warmup_time_hist.pdf')
plt.savefig(figure_path)
plt.close()

# %%
