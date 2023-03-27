# %%
import os
import os.path
import json
import numpy as np
import scipy.io
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from stan_helpers import StanMultiSessionAnalyzer, load_trajectories
import calcium_models

working_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(working_dir)

# %%
# initialize cell chain analysis
# specify a cell chain
stan_run = 'single-repeat'
cell_id = 5121
num_epochs = 10
ref_stan_run = '3'
ref_list_range = (1, 10)

# load metadata
with open('stan_run_meta.json', 'r') as f:
    stan_run_meta = json.load(f)

# get parameter names
param_mask = stan_run_meta[stan_run]['param_mask']
param_names = [calcium_models.param_names[i + 1]
               for i, mask in enumerate(param_mask) if mask == "1"]
param_names = ['sigma'] + param_names
num_params = len(param_names)

# get directories for sampled cells, as well as output of analysis
run_root = os.path.join('../../result', stan_run_meta[stan_run]['output_dir'])
session_list = [f'round-{i:02d}' for i in range(num_epochs)]
session_dirs = [os.path.join(run_root, 'samples', f'cell-{cell_id:04d}', s)
                for s in session_list]
output_dir = os.path.join(run_root, 'multi-sample-analysis')

# initialize the analyzer
analyzer = StanMultiSessionAnalyzer(session_list, output_dir, session_dirs,
                                    param_names=param_names)
session_list = analyzer.session_list
num_epochs = analyzer.num_sessions

ref_run_root = os.path.join('../../result',
                               stan_run_meta[ref_stan_run]['output_dir'])
ref_output_dir = os.path.join(
    ref_run_root,
    f'multi-sample-analysis-{ref_list_range[0]:04d}-{ref_list_range[1]:04d}'
)
ref_cell_list_path = os.path.join('cell_lists',
                                  stan_run_meta[ref_stan_run]['cell_list'])
ref_cell_list = pd.read_csv(ref_cell_list_path, sep='\t')
ref_cell_ids = ref_cell_list.iloc[
    ref_list_range[0]:ref_list_range[1] + 1, 0].to_list()
ref_session_list = [str(c) for c in ref_cell_ids]
ref_session_dirs = [os.path.join(ref_run_root, 'samples', f'cell-{c:04d}')
                    for c in ref_cell_ids]
ref_analyzer = StanMultiSessionAnalyzer(session_list, ref_output_dir,
                                        session_dirs, param_names=param_names)
cell_order = ref_cell_ids.index(cell_id)

# load calcium trajectories
ode_variant = stan_run_meta[stan_run]['ode_variant']
calcium_ode = getattr(calcium_models, f'calcium_ode_{ode_variant}')
t0 = 200
t_downsample = 300
y_all, y0_all, ts = load_trajectories(
    t0, filter_type='moving_average', moving_average_window=20,
    downsample_offset=t_downsample
)
y0_cell = np.array([0, 0, 0.7, y0_all[cell_id]])
y_cell = [None, None, None, y_all[cell_id, :]]

# get similarity matrix
soptsc_vars = scipy.io.loadmat(
        '../../result/SoptSC/SoptSC_feature_100/workspace.mat')
similarity_matrix = soptsc_vars['W']

# change matplotlib font settings
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['font.size'] = 16

# %%
# get mean log posterior and standard deviation
mean_lps = [np.mean(analyzer.session_analyzers[i].log_posterior)
            for i in range(num_epochs)]
# print('Log posterior stats of all epochs (mean ± std): ', end='')
# print(f'{np.mean(mean_lps):.2f} ± {np.std(mean_lps, ddof=1):.2f}')
ref_mixed_chains = ref_analyzer.session_analyzers[cell_order].get_mixed_chains()
ref_mean_lps = \
    ref_analyzer.session_analyzers[cell_order].get_mean_log_posteriors()
ref_mean_lps = np.mean(
    [lp for i, lp in enumerate(ref_mean_lps) if i in ref_mixed_chains]
)

plt.figure(figsize=(5, 3), dpi=300)
plot_markersize = 10
plt.plot(mean_lps, '.', color='C0', markersize=plot_markersize)
plt.axhline(y=ref_mean_lps, color='C1')
plt.title(f'Cell {cell_id:04d}')
plt.xlabel('Epoch')
plt.xticks(ticks=np.arange(num_epochs),
           labels=[i + 1 for i in range(num_epochs)])
plt.ylabel('Mean log posterior')
plt.tight_layout()
figure_path = os.path.join(output_dir, f'cell_{cell_id:04d}_lps.pdf')
plt.savefig(figure_path)
figure_path = os.path.join(output_dir, f'cell_{cell_id:04d}_lps.png')
plt.savefig(figure_path)
plt.close()

# %%
# plot mean distances between true and simulated trajectories
traj_dists = [
    al.get_trajectory_distance(calcium_ode, 0, ts, y0_cell, y_cell, 3)
    for al in analyzer.session_analyzers
]

ref_traj_dists_path = os.path.join(ref_output_dir,
                                   'mean_trajectory_distances.csv')
ref_traj_dists = pd.read_csv(ref_traj_dists_path, index_col=0, header=None)
ref_traj_dists = ref_traj_dists.to_numpy().squeeze()

plt.figure(figsize=(5, 3), dpi=300)
plot_markersize = 10
plt.plot(traj_dists, '.', color='C0', markersize=plot_markersize)
plt.axhline(y=ref_traj_dists[cell_order], color='C1')
plt.title(f'Cell {cell_id:04d}')
plt.xlabel('Epoch')
plt.xticks(ticks=np.arange(num_epochs),
           labels=[i + 1 for i in range(num_epochs)])
plt.ylabel('Mean error')
plt.ylim((0.0, 1.0))
plt.tight_layout()
figure_path = os.path.join(output_dir, f'cell_{cell_id:04d}_traj_dists.pdf')
plt.savefig(figure_path)
figure_path = os.path.join(output_dir, f'cell_{cell_id:04d}_traj_dists.png')
plt.savefig(figure_path)
plt.close()

# %%
# make a separate legend for plot of mean trajectory distances
import matplotlib.patches as mpatches

plt.figure(figsize=(5, 0.5), dpi=300)
plt.axis('off')
legend_patches = [
    mpatches.Patch(color='C0', label='Multiple epochs'),
    mpatches.Patch(color='C1', label=stan_run_meta[ref_stan_run]['pub_name'])
]
plt.legend(
    legend_patches,
    ['Multiple epochs', stan_run_meta[ref_stan_run]['pub_name']],
    loc='center', frameon=False, bbox_to_anchor=(0.5, 0.5), ncol=2
)
plt.tight_layout()
figure_path = os.path.join(output_dir, 'traj_dist_legend.pdf')
plt.savefig(figure_path)
plt.close()

# %%
# plot trajectory distance for reference run
plt.figure(figsize=(5, 4))
plt.plot(ref_traj_dists, '.', color='C0', markersize=plot_markersize)
plt.title(stan_run_meta[ref_stan_run]['pub_name'])
plt.xlabel('Cells')
plt.xticks(ticks=np.arange(num_epochs), labels=ref_cell_ids, rotation=90)
plt.ylabel('Mean error')
plt.ylim((0.0, 1.5))
plt.tight_layout()
figure_path = os.path.join(output_dir, 'ref_traj_dists.pdf')
plt.savefig(figure_path)
plt.close()

# %%
# plot marginal distribution of all epochs vs from cell chain
sample_list = [ref_analyzer.session_analyzers[cell_order].get_samples()] + \
    [al.get_samples() for al in analyzer.session_analyzers]
for i, sample in enumerate(sample_list):
    sample.columns = param_names
    if i == 0:
        sample.insert(0, 'Epoch', 'Similar')
    else:
        sample.insert(0, 'Epoch', i - 1)
all_samples = pd.concat(sample_list, ignore_index=True)

xtick_locs = np.arange(len(sample_list))
xtick_labels = ['Similar'] + [i + 1 for i in range(num_epochs)]
figure_path = os.path.join(output_dir, f'cell_{cell_id:04d}_params.pdf')
with PdfPages(figure_path) as pdf:
    for param in param_names:
        plt.figure(figsize=(5, 3), dpi=300)
        sns.boxplot(data=all_samples, x='Epoch', y=param)
        plt.xticks(ticks=xtick_locs, labels=xtick_labels, rotation=90)
        plt.ylabel(calcium_models.params_on_plot[param])
        plt.title(f'Cell {cell_id:04d}')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
