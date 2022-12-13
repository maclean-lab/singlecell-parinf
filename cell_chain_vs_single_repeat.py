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
cell_id = 5082
num_rounds = 10
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
session_list = [f'round-{i:02d}' for i in range(num_rounds)]
session_dirs = [os.path.join(run_root, 'samples', f'cell-{cell_id:04d}', s)
                for s in session_list]
output_dir = os.path.join(run_root, 'multi-sample-analysis')

# initialize the analyzer
analyzer = StanMultiSessionAnalyzer(session_list, output_dir, session_dirs,
                                    param_names=param_names)
session_list = analyzer.session_list
num_rounds = analyzer.num_sessions

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
            for i in range(num_rounds)]
print('Log posterior stats of all rounds (mean ± std): ', end='')
print(f'{np.mean(mean_lps):.2f} ± {np.std(mean_lps, ddof=1):.2f}')

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

plt.figure(figsize=(6, 4), dpi=300)
plot_markersize = 10
plt.plot(traj_dists, '.', markersize=plot_markersize,
         label='Current cell (self repeat)')
plt.plot(ref_traj_dists, '.', markersize=plot_markersize,
         label=f'Other cells in {stan_run_meta[ref_stan_run]["pub_name"]}')
plt.plot(cell_order, ref_traj_dists[cell_order], '.',
         markersize=plot_markersize,
         label=f'Current cell in {stan_run_meta[ref_stan_run]["pub_name"]}')
plt.xlabel('Round')
plt.xticks(ticks=np.arange(num_rounds),
           labels=[i + 1 for i in range(num_rounds)])
plt.ylabel('Mean trajectory distance')
plt.ylim((0.0, 1.5))
plt.legend()
plt.tight_layout()
figure_path = os.path.join(output_dir, f'cell_{cell_id:04d}_traj_dists.pdf')
plt.savefig(figure_path)
figure_path = os.path.join(output_dir, f'cell_{cell_id:04d}_traj_dists.png')
plt.savefig(figure_path)
plt.close()

# %%
# plot marginal distribution of all rounds vs from cell chain
sample_list = [ref_analyzer.session_analyzers[cell_order].get_samples()] + \
    [al.get_samples() for al in analyzer.session_analyzers]
for i, sample in enumerate(sample_list):
    sample.columns = param_names
    if i == 0:
        sample.insert(0, 'Round', 'Similar')
    else:
        sample.insert(0, 'Round', i - 1)
all_samples = pd.concat(sample_list, ignore_index=True)

xtick_locs = np.arange(len(sample_list))
xtick_labels = ['Similar'] + [i + 1 for i in range(num_rounds)]
figure_path = os.path.join(output_dir, f'cell_{cell_id:04d}_params.pdf')
with PdfPages(figure_path) as pdf:
    for param in param_names:
        plt.figure(figsize=(6, 4), dpi=300)
        sns.boxplot(data=all_samples, x='Round', y=param)
        plt.xticks(ticks=xtick_locs, labels=xtick_labels, rotation=90)
        plt.ylabel(param)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
