# %%
import os
import os.path
import sys
import itertools
import math
import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm

from stan_helpers import StanSessionAnalyzer, StanMultiSessionAnalyzer, \
    load_trajectories, simulate_trajectory, get_mode_continuous_rv
import calcium_models

working_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(working_dir)

# %%
# define metadata for analysis
stan_run_meta = {}
stan_run_meta['-1'] = {'param_mask': '0111111111111111111',
                       'ode_variant': 'equiv_2', 'pub_name': 'Full-1'}
stan_run_meta['-2'] = {'param_mask': '0111111111111111111',
                       'ode_variant': 'equiv_2', 'pub_name': 'Full-2'}
stan_run_meta['-3'] = {'param_mask': '0111111111111111111',
                       'ode_variant': 'equiv_2', 'pub_name': 'Full-3'}
stan_run_meta['-3-1.0'] = {'param_mask': '0111111111111111111',
                       'ode_variant': 'equiv_2', 'pub_name': 'Scale-1.0'}
stan_run_meta['-3-2.0'] = {'param_mask': '0111111111111111111',
                       'ode_variant': 'equiv_2', 'pub_name': 'Scale-2.0'}
stan_run_meta['-simple-prior'] = {'param_mask': '0111111111111111111',
                                  'ode_variant': 'equiv_2', 'pub_name': ''}
stan_run_meta['-const-eta1'] = {'param_mask': '0111111111111101111',
                                'ode_variant': 'equiv_2_const_eta1',
                                'pub_name': 'Red-eta1'}
stan_run_meta['-const-Be'] = {'param_mask': '0111111111011111111',
                              'ode_variant': 'equiv_2_const_Be',
                              'pub_name': 'Red-B_e'}
stan_run_meta['-const-Be-eta1'] = {'param_mask': '0111111111011101111',
                                   'ode_variant': 'equiv_2_const_Be_eta1',
                                   'pub_name': 'Red-B_e-eta1'}
stan_run_meta['-lemon-prior-1000'] = {'param_mask': '0111111111111111111',
                                      'ode_variant': 'equiv_2',
                                      'pub_name': 'Idv-cells-1000'}
stan_run_meta['-lemon-prior-500'] = {'param_mask': '0111111111111111111',
                                      'ode_variant': 'equiv_2',
                                      'pub_name': 'Idv-cells-500'}

root_cell_id = 5106
first_cell_order = 1
last_cell_order = 500
# stan_run_suffix = '-1'
# stan_run_suffix = '-2'
# stan_run_suffix = '-3'
# stan_run_suffix = '-3-1.0'
# stan_run_suffix = '-3-2.0'
# stan_run_suffix = '-simple-prior'
# stan_run_suffix = '-const-eta1'
# stan_run_suffix = '-const-Be'
stan_run_suffix = '-const-Be-eta1'
# stan_run_suffix = '-lemon-prior-1000'
# stan_run_suffix = '-lemon-prior-500'

# additional flags
simulate_trajectories = False
sensitive_params = ['L', 'Katp', 'KoffPLC', 'Vplc', 'Kip3', 'KoffIP3', 'eta2',
                    'd1', 'a', 'epr', 'dinh']
pca_sampled_only = False
use_custom_xticks = True

# get parameter names
all_param_names = ['sigma', 'KonATP', 'L', 'Katp', 'KoffPLC', 'Vplc', 'Kip3',
                   'KoffIP3', 'a', 'dinh', 'Ke', 'Be', 'd1', 'd5', 'epr',
                   'eta1', 'eta2', 'eta3', 'c0', 'k3']
param_mask = stan_run_meta[stan_run_suffix]['param_mask']
param_names = [all_param_names[i + 1] for i, mask in enumerate(param_mask)
               if mask == "1"]
param_names = ['sigma'] + param_names
num_params = len(param_names)
# select_params = 'd5', 'eta2', 'KoffIP3', 'k3', 'epr', 'KoffPLC', 'Katp'
select_param_pairs = [('KoffPLC', 'Katp'), ('eta3', 'c0'), ('epr', 'eta2'),
                      ('a', 'dinh'), ('KoffPLC', 'a')]

# get cell list
cell_list_path = f'cell_lists/dfs_feature_100_root_{root_cell_id}_0.000_1.8.txt'
full_cell_list = pd.read_csv(cell_list_path, sep='\t')
cell_list = full_cell_list.iloc[first_cell_order:last_cell_order + 1, :]

# get directories for sampled cells, as well as output of analysis
output_root = '../../result/stan-calcium-model-'
if stan_run_suffix.startswith('-lemon'):
    output_root = f'{output_root}equiv_2{stan_run_suffix}'
else:
    output_root = f'{output_root}100-root-{root_cell_id}{stan_run_suffix}'
output_dir = f'multi-sample-analysis-{first_cell_order:04d}' + \
    f'-{last_cell_order:04d}'
if pca_sampled_only:
    output_dir += '-pca-sampled-only'
output_dir = os.path.join(output_root, output_dir)
session_list = [str(c) for c in cell_list['Cell']]
session_dirs = [f'samples/cell-{c:04d}' for c in cell_list['Cell']]
session_dirs = [os.path.join(output_root, sd) for sd in session_dirs]

# load calcium trajectories
ode_variant = stan_run_meta[stan_run_suffix]['ode_variant']
calcium_ode = getattr(calcium_models, f'calcium_ode_{ode_variant}')
y, y0, ts = load_trajectories(200, filter_type='moving_average',
    moving_average_window=20, downsample_offset=300)

# %%
# initialize the analyzer for the cell chain
print('Initializing the analyzer for the cell chain...')
analyzer = StanMultiSessionAnalyzer(session_list, output_dir, session_dirs,
                                    param_names=param_names)
mixed_cells = [int(c) for c in analyzer.session_list]
num_mixed_cells = analyzer.num_sessions

# initialize the analyzer for root cell
print('Initializing the analyzer for root cell...')
root_output_dir = os.path.join(
    '../../result', 'stan-calcium-model-equiv_2-lemon-prior-1000', 'samples',
    'cell-5106')
root_analyzer = StanSessionAnalyzer(root_output_dir, param_names=param_names)

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
y0_run[:, 3] = y0[mixed_cells]
y_run = y[mixed_cells, :]
analyzer.simulate_trajectories(calcium_ode, 0, ts, y0_run, y_run, 3,
                               output_paths=figure_paths)

# %%
# simulate trajectories using Lemon prior
from matplotlib.backends.backend_pdf import PdfPages

def plot_lemon_trajectories(num_rows=4, num_cols=2):
    num_subplots_per_page = num_rows * num_cols
    num_plots = num_mixed_cells
    num_pages = math.ceil(num_plots / num_subplots_per_page)

    with PdfPages(lemon_traj_path) as pdf:
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
                cell_idx = page * num_subplots_per_page + plot_idx
                plt.subplot(num_rows, num_cols, plot_idx + 1)
                plt.plot(ts, y_lemon[cell_idx, :])

                plt.title(mixed_cells[cell_idx])

            plt.tight_layout()
            pdf.savefig()
            plt.close()

lemon_prior_spec_path = os.path.join('stan_models', ode_variant,
                                     'calcium_model_alt_prior.txt')
lemon_prior_spec = pd.read_csv(lemon_prior_spec_path, delimiter='\t',
                               index_col=0)
lemon_prior = lemon_prior_spec['mu'].to_numpy()

y_lemon = np.empty((num_mixed_cells, ts.size))
for i, cell in enumerate(mixed_cells):
    y_cell = simulate_trajectory(calcium_ode, lemon_prior, 0, ts,
                                 np.array([0, 0, 0.7, y0[cell]]))
    y_lemon[i, :] = y_cell[:, 3]

lemon_traj_path = os.path.join(
    '../../result', 'trajectories',
    'dfs_feature_100_root_5106_0.000_1.8_lemon_prior.pdf')
plot_lemon_trajectories()

# %%
# sensitivity test
sensitivity_dir = os.path.join(output_root, 'sensitivity-test')
if not os.path.exists(sensitivity_dir):
    os.mkdir(sensitivity_dir)

figure_paths = []
for i, cell in enumerate(mixed_cells):
    cell_order = first_cell_order + i
    chain_fp = os.path.join(sensitivity_dir, f'{cell_order:04d}_cell_{cell}')
    figure_paths.append(chain_fp)

y0_run = np.zeros((num_mixed_cells, 4))
y0_run[:, 2] = 0.7
y0_run[:, 3] = y0[mixed_cells]
y_run = y[mixed_cells, :]
sensitivity_dists = analyzer.run_sensitivity_test(
    calcium_ode, 0, ts, y0_run, y_run, 3, sensitive_params, figure_size=(6, 4),
    output_path_prefixes=figure_paths)
sensitivity_dists.to_csv(os.path.join(sensitivity_dir, 'sensitivity_dists.csv'))

# %%
# gather stats for trajectory distances from sensitivity test
sensitivity_dist_stats = pd.DataFrame(index=sensitivity_dists.columns,
                                      columns=['Mean', 'Std', 'Mode'])

for col_name, col in sensitivity_dists.items():
    # calculate stats for the param-percentile combo
    sensitivity_dist_stats.loc[col_name, 'Mean'] = col.mean()
    sensitivity_dist_stats.loc[col_name, 'Std'] = col.std()
    sensitivity_dist_stats.loc[col_name, 'Mode'] = \
        get_mode_continuous_rv(col, method='histogram')

    # make a histogram for trajectory distances
    figure_path = os.path.join(sensitivity_dir, f'{col_name}_hist.pdf')
    plt.figure(figsize=(11, 8.5), dpi=300)
    plt.hist(col, bins=20)
    plt.savefig(figure_path)
    plt.close()

sensitivity_dist_stats.to_csv(
    os.path.join(sensitivity_dir, 'sensitivity_dist_stats.csv'))

# %%
# make a heatmap for modes of trajectory distances
sensitivity_dist_modes = pd.DataFrame(index=sensitive_params,
                                      columns=[0.01, 0.99], dtype=float)
for param, pct in itertools.product(sensitive_params, [0.01, 0.99]):
    sensitivity_dist_modes.loc[param, pct] = \
        sensitivity_dist_stats.loc[f'{param}_{pct}', 'Mode']
figure_path = os.path.join(sensitivity_dir,
                           'sensitivity_dist_modes_heatmap.pdf')
plt.figure(figsize=(8.5, 11), dpi=300)
ax = plt.gca()
heatmap = ax.imshow(sensitivity_dist_modes, vmin=0)
ax.set_xticks(np.arange(2))
ax.set_xticklabels([0.01, 0.99])
ax.set_yticks(np.arange(len(sensitive_params)))
ax.set_yticklabels(sensitive_params)
# draw text on each block
for i, j in itertools.product(range(len(sensitive_params)), range(2)):
    ax.text(j, i, f'{sensitivity_dist_modes.iloc[i, j]:.2f}', ha='center',
            va='center', color='w')
plt.colorbar(heatmap, shrink=0.3)
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
        calcium_ode, 0, ts, np.array([0, 0, 0.7, y0[mixed_cells[0]]]),
        subsample_step_size=50, plot=False, verbose=False)
    for chain in range(analyzer.num_chains):
        figure_path = os.path.join(
            traj_pred_dir, f'0001_cell_{mixed_cells[0]}_{chain}.pdf')
        fig = Figure(figsize=(3, 2), dpi=300)
        ax = fig.gca()
        ax.plot(ts, y[mixed_cells[0], :], 'ko', fillstyle='none')
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
y0_run[:, 3] = y0[next_cells]
y_run = y[next_cells, :]
analyzer.simulate_trajectories(calcium_ode, 0, ts, y0_run, y_run, 3,
                               exclude_sessions=[str(mixed_cells[-1])],
                               output_paths=figure_paths)

# %%
# plot trajectory from sample mean for sampling vs prediction
for i, cell in enumerate(mixed_cells):
    # simulate trajectory using sample mean
    y_sample_mean = \
        analyzer.session_analyzers[i].get_sample_mean_trajectory(
            calcium_ode, 0, ts, np.array([0, 0, 0.7, y0[cell]]))

    # simulate trajectory using sample mean of previous cell
    if i == 0:
        y_sample_mean_pred = root_analyzer.get_sample_mean_trajectory(
                calcium_ode, 0, ts, np.array([0, 0, 0.7, y0[cell]]))
    else:
        y_sample_mean_pred = \
            analyzer.session_analyzers[i - 1].get_sample_mean_trajectory(
                calcium_ode, 0, ts, np.array([0, 0, 0.7, y0[cell]]))

    # plot the trajectories
    output_path = os.path.join(output_root, 'prediction-by-predecessor',
                               f'{i:04d}_cell_{cell:04d}_vs_sampled.pdf')
    plt.figure(figsize=(11, 8.5), dpi=300)
    plt.plot(ts, y[cell, :], label='True')
    plt.plot(ts, y_sample_mean[:, 3], label='Sampled')
    plt.plot(ts, y_sample_mean_pred[:, 3], label='Predicted')
    plt.legend()
    plt.savefig(output_path)
    plt.close()

# %%
# get trajectory distances if predicted by sample from predecessors
traj_pred_dists = np.empty(num_mixed_cells)
# root cell
traj_pred_dists[0] = root_analyzer.get_trajectory_distance(
    calcium_ode, 0, ts, np.array([0, 0, 0.7, y0[mixed_cells[0]]]),
    y[mixed_cells[0], :], 3)

# all other cells
for i, cell in enumerate(mixed_cells[1:]):
    traj_pred_dists[i + 1] = \
        analyzer.session_analyzers[i].get_trajectory_distance(
            calcium_ode, 0, ts, np.array([0, 0, 0.7, y0[cell]]), y[cell, :], 3)

traj_dist_path = os.path.join(output_dir, 'mean_trajectory_distances.csv')
traj_dists = pd.read_csv(traj_dist_path, index_col=0, header=None,
                         squeeze=True)

# %%
# histogram of distance from predicted to true
traj_pred_dist_path = os.path.join(
    output_dir, 'mean_predicted_trajectory_distance.pdf')
plt.figure(figsize=(11, 8.5), dpi=300)
plt.hist(traj_pred_dists, bins=25, range=(-50, 50))
plt.savefig(traj_pred_dist_path)
plt.close()

# violin plot for sampled vs predicted
traj_dist_compare_path = os.path.join(
    output_dir, 'mean_trajectory_distances_sampled_vs_prediction.pdf')
plt.figure(figsize=(11, 8.5), dpi=300)
plt.violinplot([traj_dists, traj_pred_dists])
plt.xticks(ticks=[1, 2], labels=['From posterior', 'From prior'])
plt.savefig(traj_dist_compare_path)
plt.close()

# histogram of sampled vs predicted
traj_dist_diffs = traj_pred_dists - traj_dists
traj_dist_diff_path = os.path.join(
    output_dir, 'mean_trajectory_distances_sampled_vs_prediction_diff.pdf')
plt.figure(figsize=(11, 8.5), dpi=300)
plt.hist(traj_dist_diffs, bins=25, range=(-50, 50))
plt.savefig(traj_dist_diff_path)
plt.close()

# %%
# predict unsampled cells using similar cells
rng_seed = 0
bit_generator = np.random.MT19937(rng_seed)
rng = np.random.default_rng(bit_generator)
soptsc_vars = scipy.io.loadmat(
        "../../result/SoptSC/SoptSC_feature_100/workspace.mat")
similarity_matrix = soptsc_vars["W"]
num_unsampled = 50
unsampled_cells = full_cell_list.loc[last_cell_order + 1:, 'Cell']
unsampled_cells = unsampled_cells.sample(n=num_unsampled,
                                         random_state=bit_generator)
unsampled_pred_dir = os.path.join(
    output_root, f'prediction-unsampled-{num_unsampled}-{rng_seed}')
if not os.path.exists(unsampled_pred_dir):
    os.mkdir(unsampled_pred_dir)

def predict_unsampled(num_sampled, use_similar_cell=True):
    traj_dist_table = pd.DataFrame(index=range(num_unsampled),
                                   columns=['Cell', 'SampledCell', 'Distance'])

    sampled_cell_list = cell_list.loc[:num_sampled, 'Cell'].to_numpy()
    num_zero_sim = 0
    for i, cell in enumerate(tqdm(unsampled_cells)):
        # find the most similar cell among sampled ones
        if use_similar_cell:
            cells_by_similarity = np.argsort(similarity_matrix[cell, :])[::-1]
            sampled_cell = next(c for c in cells_by_similarity
                                if c != cell and c in sampled_cell_list)
        else:
            sampled_cell = rng.choice(sampled_cell_list)

        if similarity_matrix[cell, sampled_cell] == 0:
            num_zero_sim += 1

        # predict using the most similar cell
        sampled_cell_order = cell_list['Cell'].to_list().index(sampled_cell)
        sampled_cell_analyzer = analyzer.session_analyzers[sampled_cell_order]

        # plot predicted trajectories
        traj_pred = sampled_cell_analyzer.simulate_chains(
            calcium_ode, 0, ts, np.array([0, 0, 0.7, y0[cell]]),
            subsample_step_size=50, plot=False, verbose=False)
        mixed_chains = sampled_cell_analyzer.get_mixed_chains()
        traj_pred_mixed = np.concatenate(
            [traj_pred[c][:, : ,3] for c in mixed_chains])

        traj_pred_path = os.path.join(
            unsampled_pred_dir,
            f'{cell:04d}_{num_sampled}_{sampled_cell:04d}.pdf')
        plt.figure()
        plt.plot(ts, traj_pred_mixed.T, color='C0')
        plt.plot(ts, y[cell, :], 'ko', fillstyle='none')
        plt.savefig(traj_pred_path)
        plt.close()

        # get trajectory distances
        traj_dist = sampled_cell_analyzer.get_trajectory_distance(
            calcium_ode, 0, ts, np.array([0, 0, 0.7, y0[cell]]), y[cell, :], 3)
        traj_dist_table.loc[i, 'Cell'] = cell
        traj_dist_table.loc[i, 'SampledCell'] = sampled_cell
        traj_dist_table.loc[i, 'Distance'] = traj_dist

    print(num_zero_sim)
    sys.stdout.flush()

    return traj_dist_table

unsampled_traj_dists_similar = {}
unsampled_traj_dists_random = {}
num_sampled_for_prediction = [500, 400, 300, 200, 100]
for n in num_sampled_for_prediction:
    unsampled_traj_dists_similar[n] = predict_unsampled(n)
    unsampled_traj_dists_similar_path = os.path.join(
        unsampled_pred_dir,
        f'mean_trajectory_distances_prediction_unsampled_similar_{n}.csv')
    unsampled_traj_dists_similar[n].to_csv(unsampled_traj_dists_similar_path)

    unsampled_traj_dists_random[n] = predict_unsampled(
        n, use_similar_cell=False)
    unsampled_traj_dists_random_path = os.path.join(
        unsampled_pred_dir,
        f'mean_trajectory_distances_prediction_unsampled_random_{n}.csv')
    unsampled_traj_dists_similar[n].to_csv(unsampled_traj_dists_random_path)

# %%
from scipy.stats import ks_2samp
unsampled_traj_dists_ks_stats = pd.DataFrame(columns=['Stat', 'p-value'],
                                             index=num_sampled_for_prediction)
for n in num_sampled_for_prediction:
    ks, p = ks_2samp(unsampled_traj_dists_similar[n]['Distance'],
                     unsampled_traj_dists_random[n]['Distance'])
    unsampled_traj_dists_ks_stats.loc[n, 'Stat'] = ks
    unsampled_traj_dists_ks_stats.loc[n, 'p-value'] = p

unsampled_traj_dists_ks_stats_path = os.path.join(
    unsampled_pred_dir, 'mean_trajectory_distances_similar_vs_random_ks.csv')
unsampled_traj_dists_ks_stats.to_csv(unsampled_traj_dists_ks_stats_path)

# %%
for n in num_sampled_for_prediction:
    # make histogram of trajectory distances from prediction of unsampled cells
    figure_path = os.path.join(
        unsampled_pred_dir,
        f'mean_trajectory_distances_prediction_unsampled_from_similar_{n}.pdf')
    plt.figure(figsize=(11, 8.5), dpi=300)
    plt.hist(unsampled_traj_dists_similar[n]['Distance'], bins=20)
    plt.savefig(figure_path)
    plt.close()

    figure_path = os.path.join(
        unsampled_pred_dir,
        f'mean_trajectory_distances_prediction_unsampled_from_random_{n}.pdf')
    plt.figure(figsize=(11, 8.5), dpi=300)
    plt.hist(unsampled_traj_dists_random[n]['Distance'], bins=20)
    plt.savefig(figure_path)
    plt.close()

    figure_path = os.path.join(
        unsampled_pred_dir,
        f'mean_trajectory_distances_prediction_unsampled_{n}.pdf')
    plt.figure(figsize=(11, 8.5), dpi=300)
    plt.hist(unsampled_traj_dists_similar[n]['Distance'], bins=20,
             range=(0, 40), alpha=0.5, label='From similar cell')
    plt.hist(unsampled_traj_dists_random[n]['Distance'], bins=20,
             range=(0, 40), alpha=0.5, label='From random cell')
    plt.legend()
    plt.savefig(figure_path)
    plt.close()

    # make histogram of similarity
    figure_path = os.path.join(unsampled_pred_dir,
                               f'unsampled_similar_similarity_{n}.pdf')
    plt.figure(figsize=(11, 8.5), dpi=300)
    similarity = [similarity_matrix[row['Cell'], row['SampledCell']]
                  for _, row in unsampled_traj_dists_similar[n].iterrows()]
    plt.hist(similarity)
    plt.savefig(figure_path)
    plt.close()

    figure_path = os.path.join(unsampled_pred_dir,
                               f'unsampled_random_similarity_{n}.pdf')
    plt.figure(figsize=(11, 8.5), dpi=300)
    similarity = [similarity_matrix[row['Cell'], row['SampledCell']]
                  for _, row in unsampled_traj_dists_random[n].iterrows()]
    plt.hist(similarity)
    plt.savefig(figure_path)
    plt.close()

# %%
# make plots for basic stats
print('Plotting sampling time...')
analyzer.plot_sampling_time(time_unit='m', xticks=xticks)
print('Plotting mean tree depths...')
analyzer.plot_mean_tree_depths(tree_depth_min=0, tree_depth_max=15,
                               xticks=xticks)
print('Plotting mean log posteriors...')
analyzer.plot_mean_log_posteriors(xticks=xticks)
print('Plotting R^hat of posterior')
analyzer.plot_posterior_rhats(xticks=xticks)
print('Plotting mean distances between true and simulated trajectories...')
analyzer.plot_mean_trajectory_distances(
    calcium_ode, 0, ts, y0[mixed_cells], y[mixed_cells, :], dist_min=0,
    dist_max=50, xticks=xticks)

# %%
print('Plotting R^hats vs mean trajectory distances...')
lp_rhats_vs_traj_dists_stats = analyzer.plot_lp_rhats_vs_trajectory_distances(
        calcium_ode, 0, ts, y0[mixed_cells], y[mixed_cells, :])
print('Plotting mean log posteriors vs mean trajectory distances...')
mean_lps_vs_traj_dists_stats = analyzer.plot_mean_lps_vs_trajectory_distances(
        calcium_ode, 0, ts, y0[mixed_cells], y[mixed_cells, :])
print('Plotting mean log posteriors vs R^hats...')
mean_lps_vs_lp_rhats = analyzer.plot_mean_lps_vs_lp_rhats()

# %%
param_plot_titles = [stan_run_meta[stan_run_suffix]['pub_name']] * num_params
print('Making violin plot of sampled parameters...')
analyzer.plot_parameter_violin(page_size=(6, 2), num_rows=1, num_cols=1,
                               xticks=xticks, titles=param_plot_titles,
                               y_labels=param_names)
print('Making ribbon plot of sampled parameters...')
analyzer.plot_parameter_ribbon(page_size=(6, 2), num_rows=1, num_cols=1)
print('Making box plot of sampled parameters...')
analyzer.plot_parameter_box(page_size=(6, 2), num_rows=1, num_cols=1,
                            xticks=xticks)

# %%
print('Plotting select pairs of parameters...')
session_param_pairs_dir = os.path.join(output_root, 'param-pairs')
if not os.path.exists(session_param_pairs_dir):
    os.mkdir(session_param_pairs_dir)

output_path_prefixes = [
    os.path.join(session_param_pairs_dir, f'{idx:04d}_cell_{session}')
    for idx, session in enumerate(analyzer.session_list)]
analyzer.plot_param_pairs_all_sessions(
    select_param_pairs, output_path_prefixes=output_path_prefixes)

# %%
analyzer.get_sample_means()

# %%
print('Plotting select pairs of parameters...')
# param_pair_sessions = analyzer.session_list[::50].tolist()
param_pair_sessions =['5121', '5104', '4996', '4962', '4918', '4824', '4800',
                      '4881', '4531', '4571']
for pairs in select_param_pairs:
    analyzer.plot_param_pairs(pairs, sessions=param_pair_sessions)

# %%
print('Loading gene expression data and preprocessing...')
analyzer.load_expression_data('../../data/vol_adjusted_genes_transpose.txt')
print('Filtering sessions with extreme samples...')
analyzer.filter_sessions(z_score_max=3.0)
print('Plotting correlation between sampled parameters...')
analyzer.get_parameter_correlations(plot=True)

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
analyzer.plot_select_genes_vs_params(
    high_corr_pairs, regressors_trained, 'high_corr_pairs_scatter_huber.pdf',
    figure_size=(8, 6), num_rows=1, num_cols=1)

print('All done!')

# %%
