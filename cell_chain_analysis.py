# %%
import os
import os.path
import json
import numpy as np
import scipy.io
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from stan_helpers import StanMultiSessionAnalyzer, load_trajectories
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
stan_runs = ['const-Be-eta1']
# stan_runs = ['const-Be-eta1-mixed-4']
# stan_runs = [f'const-Be-eta1-mixed-{i}' for i in range(5)]
# stan_runs = ['const-Be-eta1-random-1']
# stan_runs = [f'const-Be-eta1-random-{i}' for i in range(1, 7)]
# stan_run = 'lemon-prior-1000'
# stan_run = 'lemon-prior-500'

# additional flags
num_runs = len(stan_runs)
list_ranges = [(1, 500)]
# list_ranges = [(1, 100)] * num_runs
# list_ranges = [(1, 571)]
# list_ranges = [(1, 571), (1, 372), (1, 359), (1, 341), (1, 335), (1, 370)]
pca_sampled_only = False
use_custom_xticks = True

# load metadata
with open('stan_run_meta.json', 'r') as f:
    stan_run_meta = json.load(f)

# get parameter names
param_mask = stan_run_meta[stan_runs[0]]['param_mask']
param_names = [calcium_models.param_names[i + 1]
               for i, mask in enumerate(param_mask) if mask == "1"]
param_names = ['sigma'] + param_names
num_params = len(param_names)
select_param_pairs = [('KoffPLC', 'Katp'), ('eta3', 'c0'), ('epr', 'eta2'),
                      ('a', 'dinh'), ('KoffPLC', 'a')]

# get cell list
session_list = []
session_dirs = []
for run, lr in zip(stan_runs, list_ranges):
    run_root = os.path.join('../../result', stan_run_meta[run]['output_dir'])
    cell_list_path = os.path.join('cell_lists',
                                  stan_run_meta[run]['cell_list'])
    run_cell_list = pd.read_csv(cell_list_path, sep='\t')
    sampled_cells = run_cell_list.iloc[lr[0]:lr[1] + 1, 0]
    session_list.extend([str(c) for c in sampled_cells])
    session_dirs.extend([os.path.join(run_root, 'samples', f'cell-{c:04d}')
                         for c in sampled_cells])

# get directories for sampled cells, as well as output of analysis
if num_runs == 1:
    output_root = stan_run_meta[stan_runs[0]]['output_dir']
else:
    output_root = stan_run_meta[stan_runs[0]]['output_dir'][:-2] + '-all'

output_root = os.path.join('../../result', output_root)
output_dir = 'multi-sample-analysis'
if num_runs == 1:
    output_dir += f'-{list_ranges[0][0]:04d}-{list_ranges[0][1]:04d}'
if pca_sampled_only:
    output_dir += '-pca-sampled-only'
output_dir = os.path.join(output_root, output_dir)

# initialize the analyzer for the cell chain
print('Initializing the analyzer for the cell chain...')
analyzer = StanMultiSessionAnalyzer(session_list, output_dir, session_dirs,
                                    param_names=param_names)
session_list = analyzer.session_list
session_list_int = [int(c) for c in session_list]
num_sessions = analyzer.num_sessions

# load calcium trajectories
ode_variant = stan_run_meta[stan_runs[0]]['ode_variant']
calcium_ode = getattr(calcium_models, f'calcium_ode_{ode_variant}')
t0 = 200
t_downsample = 300
y_all, y0_all, ts = load_trajectories(t0, filter_type='moving_average',
    moving_average_window=20, downsample_offset=t_downsample)
y0_sessions = y0_all[session_list_int]
y_sessions = y_all[session_list_int, :]

# get similarity matrix
soptsc_vars = scipy.io.loadmat(
        '../../result/SoptSC/SoptSC_feature_100/workspace.mat')
similarity_matrix = soptsc_vars['W']

# change matplotlib font settings
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['font.size'] = 16

# set ticks on x-axis for plots
if use_custom_xticks:
    if num_sessions > 50:
        num_xticks = int(np.round(num_sessions / 20)) + 1
        xtick_locs = np.arange(num_xticks) * 20 - 1
        xtick_locs[0] += 1
        xtick_labels = xtick_locs + 1
    elif num_sessions > 10:
        num_xticks = int(np.round(num_sessions / 5)) + 1
        xtick_locs = np.arange(num_xticks) * 5
        xtick_labels = xtick_locs + 1
    else:
        num_xticks = num_sessions
        xtick_locs = np.arange(num_xticks)
        xtick_labels = xtick_locs + 1

    xticks = {'ticks': xtick_locs, 'labels': xtick_labels}
else:
    xticks = None

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

# %%
# make scatter plots for comparing basic stats
print('Plotting mean distances between true and simulated trajectories...')
analyzer.plot_mean_trajectory_distances(
    calcium_ode, 0, ts, y0_sessions, y_sessions, dist_min=0, dist_max=50,
    xticks=xticks)
print('Plotting R^hats vs mean trajectory distances...')
lp_rhats_vs_traj_dists_stats = analyzer.plot_lp_rhats_vs_trajectory_distances(
        calcium_ode, 0, ts, y0_sessions, y_sessions)
print('Plotting mean log posteriors vs mean trajectory distances...')
mean_lps_vs_traj_dists_stats = analyzer.plot_mean_lps_vs_trajectory_distances(
        calcium_ode, 0, ts, y0_sessions, y_sessions)
print('Plotting mean log posteriors vs R^hats...')
mean_lps_vs_lp_rhats = analyzer.plot_mean_lps_vs_lp_rhats()

# %%
# plot posterior samples along cell chain
param_plot_titles = [stan_run_meta[stan_runs[0]]['pub_name']] * num_params

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
print('Plotting select pairs of parameters...', flush=True)
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
param_pair_sessions = analyzer.session_list[::50].tolist()
param_plot_titles = ['MAP values'] \
    + [f'Cell {c}' for c in param_pair_sessions]
for pairs in select_param_pairs:
    analyzer.plot_param_pairs(
        pairs, sessions=param_pair_sessions, num_rows=1, num_cols=1,
        page_size=(4, 4), param_names_on_plot=calcium_models.params_on_plot,
        titles=param_plot_titles)

# %%
# make legend pair plots of parameters
plt.figure(figsize=(2, 0.5), dpi=300)
gradient = np.linspace(0, 1, 100)
gradient = gradient[np.newaxis, :]
plt.imshow(gradient, aspect=3.0, cmap=plt.get_cmap('viridis'))
plt.axis('off')
plt.title('Cell positions', fontdict={'fontsize': 'medium'})
figure_path = os.path.join(output_dir, 'param_pair_scatters_legend.pdf')
plt.tight_layout()
plt.savefig(figure_path)
plt.close()

# %%
# gene-parameter correlations
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
num_top_pairs = 450
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
plt.legend(legend_patches, ['Regression line'], loc='upper left',
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
# plot histogram of gene-param correlations
from statsmodels.stats.multitest import multipletests

sorted_gene_vs_param_corrs_path = os.path.join(
    output_dir, 'genes-vs-params', 'pearson_corrs_sorted.csv')

sorted_gene_vs_param_corrs = pd.read_csv(sorted_gene_vs_param_corrs_path,
                                         index_col=0)
reject, pval_adj, _, alpha_adj = multipletests(
    sorted_gene_vs_param_corrs['p-value'], alpha=0.05, method='bonferroni')

print('Correlation at cutoff for adjusted p-values:')
print(sorted_gene_vs_param_corrs.loc[sum(pval_adj < 0.05) - 1, 'Correlation'])

# %%
plt.figure(figsize=(6, 4), dpi=300)
bin_colors = ['C1'] * 4 + ['C0'] * 11 + ['C1'] * 5
_, _, hist_patches = plt.hist(sorted_gene_vs_param_corrs['Correlation'],
                              bins=20)
for p, c in zip(hist_patches, bin_colors):
    p.set_facecolor(c)
plt.xlabel('Correlation')
plt.ylabel('Count')
plt.title('Genes-parameter correlations')
legend_patches = [hist_patches[0], hist_patches[10]]
plt.legend(legend_patches,
           ['Adjusted\np-value<0.05', 'Adjusted\np-value≥0.05'],
           fontsize='small')
plt.tight_layout()
figure_path = os.path.join(output_dir, 'genes-vs-params',
                           'pearson_corrs_hist.pdf')
plt.savefig(figure_path)
plt.close('all')

# %%
# analyze warmup
warmup_time = pd.DataFrame(index=session_list,
                           columns=range(analyzer.num_chains))
warmup_iters = 500

for idx, a in zip(session_list, analyzer.session_analyzers):
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

figure_path = os.path.join(output_dir, 'warmup_time_hist.pdf')
plt.savefig(figure_path)
plt.close()

# %%
# plot positions of similar cells
plt.figure(figsize=(6, 6), dpi=300)
session_list_int.insert(0, 5106)  # add root cell
similar_cells = similarity_matrix[np.ix_(session_list_int, session_list_int)]
similar_cells = np.ceil(similar_cells)

plt.imshow(similar_cells, cmap='binary', interpolation='none')
plt.tight_layout()

figure_path = os.path.join(output_dir, 'similar_cell_positions.pdf')
plt.savefig(figure_path)
plt.close()

session_list_int = session_list_int[1:]  # remove root cell
