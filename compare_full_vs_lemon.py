# %%
import os
import os.path
import itertools

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from stan_helpers import StanMultiSessionAnalyzer

# change font settings
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['font.size'] = 16

# %%
first_cell_order = 1
last_cell_order = 10
rhat_upper_bound = 100.0

runs = ['full', 'lemon-500', 'lemon-1000']
num_runs = len(runs)
result_root = '../../result'
analyzer_dirs = {}
analyzer_dirs['full'] = 'stan-calcium-model-100-root-5106-3'
analyzer_dirs['lemon-1000'] = 'stan-calcium-model-equiv_2-lemon-prior-1000'
analyzer_dirs['lemon-500'] = 'stan-calcium-model-equiv_2-lemon-prior-500'

# load cell list
cell_list_path = f'cell_lists/dfs_feature_100_root_5106_0.000_1.8.txt'
cell_list = pd.read_csv(cell_list_path, sep='\t')
cell_list = cell_list.iloc[first_cell_order:last_cell_order + 1, :]
session_list = [str(c) for c in cell_list['Cell']]
num_cells = len(cell_list)

# initialize analyzer for each cell chain
analyzers = {}
for run in runs:
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

# define colors for cell chains
run_color_map = {}
run_color_map['full'] = 'C0'
run_color_map['lemon-1000'] = 'C1'
run_color_map['lemon-500'] = 'C2'

# %%
def make_heatmap(data, figure_name, max_value=500, colorbar_ticks=None,
                 colorbar_ticklabels=None):
    plt.figure(figsize=(4, 9), dpi=300)
    ax = plt.gca()
    capped_data = np.clip(data, None, max_value)
    heatmap = ax.imshow(capped_data)

    # set tick labels
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(num_runs))
    ax.set_xticklabels(labels=['Full-3', 'Idv-cells-500', 'Idv-cells-1000'],
                       rotation=45)
    # ax.set_yticks(np.arange(num_cells))
    ax.set_yticks([])
    # ax.set_yticklabels(labels=cell_list['Cell'])
    ax.set_yticklabels([])

    # draw an arrow along y-axis
    ax.set_ylabel('Cell position', labelpad=20)
    ax.arrow(-0.75, -0, 0, 9, head_width=0.1, head_length=0.1,
            length_includes_head=True, clip_on=False, color='k')

    # show values on each block
    for i, j in itertools.product(range(num_cells), range(num_runs)):
        ax.text(j, i, f'{data[i, j]:.0f}', ha='center', va='center', color='w')

    # add a color bar
    cb = plt.colorbar(heatmap, shrink=0.3)
    if colorbar_ticks is not None:
        cb.set_ticks(colorbar_ticks)
    if colorbar_ticklabels is not None:
        cb.set_ticklabels(colorbar_ticklabels)

    # export
    heatmap_path = os.path.join(
        result_root, 'stan-calcium-model-100-root-5106-comparison-3-vs-lemon',
        'cell-0001-0010', figure_name)
    plt.tight_layout()
    plt.savefig(heatmap_path, transparent=True)
    plt.close()

# %%
# compute stats for log posterior
lp_stats_columns = []
for run in runs:
    lp_stats_columns.append(f'{run} mean')
    lp_stats_columns.append(f'{run} std')
lp_stats = pd.DataFrame(columns=lp_stats_columns, index=cell_list['Cell'])
for run in runs:
    for cell, a in zip(cell_list['Cell'],
                       analyzers[run].session_analyzers):
        lp = np.array(a.log_posterior)
        lp_stats.loc[cell, f'{run} mean'] = lp.mean()
        lp_stats.loc[cell, f'{run} std'] = lp.std()

# %%
# print stats for log posterior
for cell in cell_list['Cell']:
    line = str(cell)

    for run in runs:
        mean = lp_stats.loc[cell, f'{run} mean']
        std = lp_stats.loc[cell, f'{run} std']
        line += f' & ${mean:.2f} \pm {std:.2f}$'

    line += ' \\\\'
    print(line)

# %%
# make a heatmap for log posterior means
lp_means = lp_stats.values[:, ::2].astype(np.double)
make_heatmap(lp_means, 'mean_log_posterior_heatmap.pdf',
             colorbar_ticks=[250, 300, 350, 400])

# %%
# compute stats for sampling time
sampling_time_stats_columns = []
for run in runs:
    sampling_time_stats_columns.append(f'{run} mean')
    sampling_time_stats_columns.append(f'{run} std')
sampling_time_stats = pd.DataFrame(columns=sampling_time_stats_columns,
                                   index=cell_list['Cell'])
for run in runs:
    for cell, a in zip(cell_list['Cell'],
                       analyzers[run].session_analyzers):
        st = a.get_sampling_time() / 60
        sampling_time_stats.loc[cell, f'{run} mean'] = st.mean()
        sampling_time_stats.loc[cell, f'{run} std'] = st.std()

# %%
# make a heatmap for sampling_time
sampling_time_means = sampling_time_stats.values[:, ::2].astype(np.double)
make_heatmap(sampling_time_means, 'sampling_time_heatmap.pdf',
             colorbar_ticks=[100, 300, 500],
             colorbar_ticklabels=['100', '300', '≥500'])

# %%
# load mean trajectory distances
trajectory_distances = {}
for run in runs:
    data_path = os.path.join(
        result_root, analyzer_dirs[run],
        f'multi-sample-analysis-{first_cell_order:04d}-{last_cell_order:04d}',
        'mean_trajectory_distances.csv')
    trajectory_distances[run] = pd.read_csv(data_path, index_col=0, header=None,
                                            squeeze=True)

# plot mean trajectory distances
trajectory_distance_path = os.path.join(
    result_root, 'stan-calcium-model-100-root-5106-comparison-3-vs-lemon',
    'cell-0001-0010', 'mean_trajectory_distances.pdf')
plt.figure(figsize=(8, 6), dpi=300)
for run in runs:
    plt.plot(trajectory_distances[run], color=run_color_map[run],
             marker='.', linestyle='')
plt.xlabel('Cell position')
plt.ylabel('Mean trajectory distance')
plt.savefig(trajectory_distance_path, transparent=True)
plt.close()

# %%