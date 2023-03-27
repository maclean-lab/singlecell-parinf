# %%
import os
import os.path
import itertools
import json

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# import seaborn as sns

from stan_helpers import StanMultiSessionAnalyzer

# change font settings
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['font.size'] = 16

# %%
first_cell_order = 1
last_cell_order = 10
rhat_upper_bound = 100.0

with open('stan_run_meta.json', 'r') as f:
    stan_run_meta = json.load(f)

stan_runs = ['3', 'lemon-prior-500', 'lemon-prior-1000']
num_runs = len(stan_runs)
result_root = '../../result'
result_dir = os.path.join(
    result_root, 'stan-calcium-model-100-root-5106-comparison-3-vs-lemon',
    f'cell-{first_cell_order:04d}-{last_cell_order:04d}')
analyzer_dirs = {}
run_color_map = {}
pub_names = []
for run in stan_runs:
    analyzer_dirs[run] = stan_run_meta[run]['output_dir']
    run_color_map[run] = stan_run_meta[run]['color']
    pub_names.append(stan_run_meta[run]['pub_name'])

# load cell list
cell_list_path = os.path.join('cell_lists', stan_run_meta['3']['cell_list'])
cell_list = pd.read_csv(cell_list_path, sep='\t')
cell_list = cell_list.iloc[first_cell_order:last_cell_order + 1, :]
session_list = [str(c) for c in cell_list['Cell']]
num_cells = len(cell_list)

# initialize analyzer for each cell chain
analyzers = {}
for run in stan_runs:
    session_dirs = [f'samples/cell-{c:04d}' for c in cell_list['Cell']]
    session_dirs = [os.path.join(result_root, analyzer_dirs[run], sd)
                    for sd in session_dirs]
    output_dir = os.path.join(
        result_root, analyzer_dirs[run],
        f'multi-sample-analysis-{first_cell_order:04d}' + \
            f'-{last_cell_order:04d}')
    analyzers[run] = StanMultiSessionAnalyzer(
        session_list, output_dir, session_dirs,
        rhat_upper_bound=rhat_upper_bound)

# %%
def make_heatmap(data, figure_name, max_value=500, xlabel=None,
                 colorbar_ticks=None, colorbar_ticklabels=None,
                 plot_cell_ids=False, plot_yaxis_arrow=False):
    plt.figure(figsize=(4, 9), dpi=300)
    ax = plt.gca()
    capped_data = np.clip(data, None, max_value)
    heatmap = ax.imshow(capped_data)

    # set tick labels
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(num_runs))
    ax.set_xticklabels(labels=pub_names, rotation=90, fontsize=20)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=20)
    if plot_cell_ids:
        ax.text(-1.5, -1, 'Cell ID', ha='center', va='center', color='k',
                fontsize=16)
        ax.set_yticks(np.arange(num_cells))
        ax.set_yticklabels(labels=cell_list['Cell'], fontsize=16)
    else:
        ax.set_yticks(np.arange(num_cells))
        ax.set_yticklabels([''] * num_cells)

    # draw an arrow along y-axis
    if plot_yaxis_arrow:
        ax.set_ylabel('Cell position', labelpad=20, fontsize=20)
        ax.arrow(-0.75, 0, 0, 9, head_width=0.1, head_length=0.1,
                 length_includes_head=True, clip_on=False, color='k')

    # set bounding box for the heatmap portion
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(9.5, -0.5)

    # show values on each block
    for i, j in itertools.product(range(num_cells), range(num_runs)):
        ax.text(j, i, f'{data[i, j]:.0f}', ha='center', va='center', color='w')

    # add a color bar
    cb = plt.colorbar(heatmap, shrink=0.75)
    if colorbar_ticks is not None:
        cb.set_ticks(colorbar_ticks)
    if colorbar_ticklabels is not None:
        cb.set_ticklabels(colorbar_ticklabels)

    # export
    heatmap_path = os.path.join(result_dir, figure_name)
    plt.tight_layout()
    plt.savefig(heatmap_path, transparent=True)
    plt.close()

# %%
# compute stats for log posterior
lp_stats_columns = []
for run in stan_runs:
    lp_stats_columns.append(f'{run} mean')
    lp_stats_columns.append(f'{run} std')
lp_stats = pd.DataFrame(columns=lp_stats_columns, index=cell_list['Cell'])
for run in stan_runs:
    for cell, a in zip(cell_list['Cell'],
                       analyzers[run].session_analyzers):
        lp = np.array(a.log_posterior)
        lp_stats.loc[cell, f'{run} mean'] = lp.mean()
        lp_stats.loc[cell, f'{run} std'] = lp.std()

# %%
# print stats for log posterior
for cell in cell_list['Cell']:
    line = str(cell)

    for run in stan_runs:
        mean = lp_stats.loc[cell, f'{run} mean']
        std = lp_stats.loc[cell, f'{run} std']
        line += f' & ${mean:.2f} \pm {std:.2f}$'

    line += ' \\\\'
    print(line)

# %%
# make a heatmap for log posterior means
lp_means = lp_stats.values[:, ::2].astype(np.double)
make_heatmap(lp_means, 'mean_log_posterior_heatmap.pdf',
             xlabel='Mean log posterior', colorbar_ticks=[250, 300, 350, 400],
             plot_cell_ids=True)

# %%
# compute stats for sampling time
sampling_time_stats_columns = []
for run in stan_runs:
    sampling_time_stats_columns.append(f'{run} mean')
    sampling_time_stats_columns.append(f'{run} std')
sampling_time_stats = pd.DataFrame(columns=sampling_time_stats_columns,
                                   index=cell_list['Cell'])
for run in stan_runs:
    for cell, a in zip(cell_list['Cell'],
                       analyzers[run].session_analyzers):
        st = a.get_sampling_time() / 60
        sampling_time_stats.loc[cell, f'{run} mean'] = st.mean()
        sampling_time_stats.loc[cell, f'{run} std'] = st.std()

# %%
# make a heatmap for sampling time
sampling_time_means = sampling_time_stats.values[:, ::2].astype(np.double)
make_heatmap(sampling_time_means, 'sampling_time_heatmap.pdf',
             xlabel='Time (minutes)', colorbar_ticks=[100, 300, 500],
             colorbar_ticklabels=['100', '300', '≥500'])

# %%
# load mean trajectory distances
trajectory_distances = {}
for run in stan_runs:
    data_path = os.path.join(
        result_root, analyzer_dirs[run],
        f'multi-sample-analysis-{first_cell_order:04d}-{last_cell_order:04d}',
        'mean_trajectory_distances.csv')
    trajectory_distances[run] = pd.read_csv(data_path, index_col=0, header=None,
                                            squeeze=True)

# plot mean trajectory distances
trajectory_distance_path = os.path.join(result_dir,
                                        'mean_trajectory_distances.pdf')
plt.figure(figsize=(8, 6), dpi=300)
for run in stan_runs:
    plt.plot(trajectory_distances[run], color=run_color_map[run],
             marker='.', linestyle='')
plt.xlabel('Cell position')
plt.ylabel('Mean trajectory distance')
plt.savefig(trajectory_distance_path, transparent=True)
plt.close()

# %%
