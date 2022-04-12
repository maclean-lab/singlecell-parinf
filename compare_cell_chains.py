# %%
import os
import os.path
import json
import numpy as np
import scipy.stats
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# %%
root_cell_id = 5106
first_cell_order = 1
last_cell_order = 200

# define directories for results produced by MultiSessionAnalyzers
run_group = 'full-models'
# run_group = '3-vs-const'
# run_group = 'scaling'
# run_group = '3-vs-lemon'
# run_group = 'similar-vs-mixed'

with open('stan_run_meta.json', 'r') as f:
    stan_run_meta = json.load(f)
with open('stan_run_comparison_meta.json', 'r') as f:
    stan_run_comparison_meta = json.load(f)

runs = stan_run_comparison_meta[run_group]['runs']
num_runs = len(runs)

output_root = '../../result'
run_dirs = [stan_run_meta[r]['output_dir'] for r in runs]
analyzer_dirs = [
    os.path.join(
        output_root, d,
        f'multi-sample-analysis-{first_cell_order:04d}-{last_cell_order:04d}')
    for d in run_dirs
]

cell_plot_order_type = stan_run_comparison_meta[run_group]['order_by']

output_dir = os.path.join(
    output_root,
    f'stan-calcium-model-100-root-{root_cell_id}-comparison-{run_group}',
    f'cell-{first_cell_order:04d}-{last_cell_order:04d}')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# get cell list
if cell_plot_order_type == 'cell_id':
    # assume same list for all runs
    cell_list_path = os.path.join(
        'cell_lists', stan_run_meta[runs[0]]['cell_list'])
    cell_list = pd.read_csv(cell_list_path, sep='\t')
    cell_plot_order = cell_list.iloc[first_cell_order:last_cell_order + 1, 0]
else:
    cell_plot_order = list(range(first_cell_order, last_cell_order + 1))

    cell_lists = {}
    for run in runs:
        cell_list_path = os.path.join(
            'cell_lists', stan_run_meta[run]['cell_list'])
        cell_lists[run] = pd.read_csv(cell_list_path, sep='\t')
        cell_lists[run] = cell_lists[run].iloc[
            first_cell_order:last_cell_order + 1, 0]
        cell_lists[run] = cell_lists[run].to_list()

pub_names = [stan_run_meta[r]['pub_name'] for r in runs]
pub_names = [n.replace('\\$', '$') for n in pub_names]

# define plot parameters
dpi = 300
figure_size = (6, 4)
run_colors = [stan_run_meta[r]['color'] for r in runs]

# change font settings
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['font.size'] = 18

marker_size = 4

# x-ticks for violin plot
violin_xticks = np.arange(num_runs) + 1
violin_xticklabels = pub_names
# x-ticks for plot of cells in one cell chain
num_cells = last_cell_order - first_cell_order + 1
if num_cells >= 100:
    num_run_xticks = int(np.round(num_cells / 50)) + 1
    run_xtick_locs = np.arange(num_run_xticks) * 50 - 1
    run_xtick_locs[0] += 1
    run_xtick_labels = run_xtick_locs + first_cell_order
else:
    num_run_xticks = int(np.round(num_cells / 5)) + 1
    run_xtick_locs = np.arange(num_run_xticks) * 5
    run_xtick_labels = run_xtick_locs + first_cell_order

# %%
# load data
trajectory_distances = []
log_posteriors = []
sampling_time = []
tree_depths = []
rhats = []
num_mixed_chains = []
for d in analyzer_dirs:
    # load trajectory distances
    data_path = os.path.join(d, 'mean_trajectory_distances.csv')
    run_data = pd.read_csv(data_path, index_col=0, header=None, squeeze=True)
    trajectory_distances.append(run_data)

    # load mean log posteriors
    data_path = os.path.join(d, 'mean_log_posteriors.csv')
    run_data = pd.read_csv(data_path, index_col=0)
    log_posteriors.append(run_data)

    # load sampling time
    data_path = os.path.join(d, 'sampling_time.csv')
    run_data = pd.read_csv(data_path, index_col=0)
    sampling_time.append(run_data)

    # load tree depths
    data_path = os.path.join(d, 'mean_tree_depths.csv')
    run_data = pd.read_csv(data_path, index_col=0, header=None, squeeze=True)
    tree_depths.append(run_data)

    # load R^hats and mixed chains
    data_path = os.path.join(d, 'posterior_rhats.csv')
    run_data = pd.read_csv(data_path, index_col=0)
    rhats.append(run_data['R_hat'])
    num_mixed_chains.append(
        run_data.loc[:, [str(c) for c in range(4)]].sum(axis=1))

# %%
# report stats
stats = pd.DataFrame()
def add_stats(data, data_name, axis=None):
    data_mean = []
    data_std = []

    for run_data in data:
        data_mean.append(run_data.values.mean(axis=axis))
        data_std.append(run_data.values.std())
    stats[f'{data_name} mean'] = data_mean
    stats[f'{data_name} sd'] = data_std

add_stats(tree_depths, 'Tree depth')
add_stats(sampling_time, 'Sampling time')
add_stats(rhats, 'R^hats')
add_stats(num_mixed_chains, 'Mixed chains')
add_stats(trajectory_distances, 'Trajectory distance')
stats.index = pub_names

print(stats)
stats.to_csv(os.path.join(output_dir, 'stats.csv'))

# print to latex code
for row, run in enumerate(runs):
    line = pub_names[row]
    for i in range(0, stats.shape[1], 2):
        line += f' & ${stats.iloc[row, i]:.2f} \pm {stats.iloc[row, i+1]:.2f}$'
    line += ' \\\\'
    print(line)

# %%
# convert tables for mean log posteriors to one table in long form
mean_log_posteriors_long = pd.DataFrame(columns=['Run', 'Cell', 'Chain', 'LP'])
num_rows = 0
for run, run_lp in zip(runs, log_posteriors):
    for cell_id, row in run_lp.iterrows():
        for chain, lp in row.items():
            if stan_run_comparison_meta[run_group]['order_by'] == 'cell_id':
                cell = cell_id
            else:
                cell = cell_lists[run].index(cell_id)

            long_row = {'Run': run, 'Cell': cell, 'Chain': chain, 'LP': lp}
            mean_log_posteriors_long.loc[num_rows] = long_row
            num_rows += 1

# convert tables for sampling time to one table in long form
sampling_time_long = pd.DataFrame(columns=['Run', 'Cell', 'Chain', 'Time'])
num_rows = 0
for run, run_st in zip(runs, sampling_time):
    for cell_id, row in run_st.iterrows():
        for chain, t in row.items():
            if stan_run_comparison_meta[run_group]['order_by'] == 'cell_id':
                cell = cell_id
            else:
                cell = cell_lists[run].index(cell_id)

            long_row = {'Run': run, 'Cell': cell, 'Chain': chain, 'Time': t}
            sampling_time_long.loc[num_rows] = long_row
            num_rows += 1

# %%
# make violin plot for trajectory distances
figure_path = os.path.join(output_dir, 'mean_trajectory_distances_violin.pdf')
plt.figure(figsize=figure_size, dpi=dpi)
dist_violins = plt.violinplot([td.values for td in trajectory_distances],
                               showmeans=True)
for violin, color in zip(dist_violins['bodies'], run_colors):
    violin.set_facecolor(color)
dist_violins['cmeans'].set_color('k')
dist_violins['cmins'].set_color('k')
dist_violins['cmaxes'].set_color('k')
dist_violins['cbars'].set_color('k')
plt.xticks(ticks=violin_xticks, labels=violin_xticklabels)
plt.ylabel('Mean trajectory distances')
# plt.ylim((0, 4))

plt.tight_layout()
plt.savefig(figure_path, transparent=True)
plt.close()

# %%
# plot mean log posterior densities
figure_path = os.path.join(output_dir, 'mean_log_posteriors.pdf')
plt.figure(figsize=figure_size, dpi=dpi)

# specify limits for log posterior
lp_low = 0
lp_high = 600
mean_log_posteriors_long['LP'].clip(lower=lp_low, upper=lp_high, inplace=True)

g = sns.stripplot(data=mean_log_posteriors_long, x='Cell', y='LP', hue='Run',
                  order=cell_plot_order, alpha=0.8, palette=run_colors,
                  size=marker_size)

# draw lines for means of chains
for i in range(num_runs):
    run_mean = np.full(num_cells, np.mean(log_posteriors[i].values))
    g.plot(run_mean, color=run_colors[i])

# set labels on axes
g.set_xticks(run_xtick_locs)
g.set_xticklabels(run_xtick_labels)
g.set_xlabel('Cell position')
g.set_ylabel('Mean log posterior')

# set limits
g.set_ylim(bottom=lp_low, top=lp_high)

# update legend
legend = g.legend()
for text, label in zip(legend.texts, pub_names):
    text.set_text(label)
legend.remove()

# export
plt.tight_layout()
plt.savefig(figure_path, transparent=True)
plt.close()

# %%
# plot sampling time
figure_path = os.path.join(output_dir, 'sampling_time.pdf')
plt.figure(figsize=(6, 7), dpi=dpi)

# make scatter plot, with upper bound for time
sampling_time_low = 0
sampling_time_high = 1000
sampling_time_long['Time'].clip(upper=sampling_time_high, inplace=True)
g = sns.stripplot(data=sampling_time_long, x='Cell', y='Time', hue='Run',
                  order=cell_plot_order, alpha=0.8, palette=run_colors,
                  size=marker_size)

# draw lines for means of chains
for i in range(num_runs):
    run_mean = np.full(num_cells, np.mean(sampling_time[i].values))
    g.plot(run_mean, color=run_colors[i], linewidth=5)

# set labels on axes
g.set_xticks(run_xtick_locs)
g.set_xticklabels(run_xtick_labels)
g.set_xlabel('Cell position')
g.set_ylabel('Time (minutes)')

# set limits
g.set_ylim(bottom=sampling_time_low, top=sampling_time_high)

# add title
g.set_title('Runtime comparison')

# update legend
legend = g.legend()
for text, label in zip(legend.texts, pub_names):
    text.set_text(label)
# legend.remove()

# export
plt.tight_layout()
plt.savefig(figure_path, transparent=True)
plt.close()

# %%
# make violin plot for sampling time
plt.figure(figsize=figure_size, dpi=dpi)
figure_path = os.path.join(output_dir, 'sampling_time_violin.pdf')
time_violins = plt.violinplot([t.values.flatten() for t in sampling_time],
                              showmeans=True)
for violin, color in zip(time_violins['bodies'], run_colors):
    violin.set_facecolor(color)
time_violins['cmeans'].set_color('k')
time_violins['cmins'].set_color('k')
time_violins['cmaxes'].set_color('k')
time_violins['cbars'].set_color('k')
plt.xticks(ticks=violin_xticks, labels=violin_xticklabels, rotation=45)
plt.ylim((0, 1000))
plt.ylabel('Time (minutes)')
plt.title('NUTS runtime comparison')
plt.tight_layout()
plt.savefig(figure_path)
plt.close()

# %%
# compare warmup time for similar vs random
analyzer_dirs[1] = analyzer_dirs[1].replace('500', '100')
analyzer_dirs[2] = analyzer_dirs[2].replace('500', '086')
analyzer_dirs[3] = analyzer_dirs[3].replace('500', '100')

# load warmup time data
warmup_time = []
for d in analyzer_dirs:
    data_path = os.path.join(d, 'warmup_time.csv')
    run_data = pd.read_csv(data_path, index_col=0)
    warmup_time.append(run_data)

# run K-S test for similar vs each random run
for i in range(1, 4):
    wt_similar = warmup_time[0].to_numpy().flatten()
    wt_similar = wt_similar[np.isfinite(wt_similar)]
    wt_random = warmup_time[i].to_numpy().flatten()
    wt_random = wt_random[np.isfinite(wt_random)]
    stat, p_val = scipy.stats.ks_2samp(wt_similar, wt_random,
                                       alternative='less')
    print(f'{stat:.8f} {p_val:.8f}')

# %%
# make legends
import matplotlib.patches as mpatches

dummy_fig = plt.figure(figsize=(3, num_runs * 0.5), dpi=dpi)
legend_patches = [mpatches.Patch(color=c, label=r)
                  for c, r in zip(run_colors, pub_names)]
dummy_fig.legend(legend_patches, pub_names, loc='center')
dummy_fig.savefig(os.path.join(output_dir, 'figure_legend.pdf'))
plt.close()
