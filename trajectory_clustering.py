# %%
import os
import os.path
import json

import numpy as np
import scipy.io
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

from stan_helpers import load_trajectories

# %%
# stan_runs = ['3']
# stan_runs = ['const-Be-eta1']
# stan_runs = ['const-Be-eta1-mixed-1']
# stan_runs = [f'const-Be-eta1-mixed-{i}' for i in range(5)]
# stan_runs = ['const-Be-eta1-random-1']
# stan_runs = [f'const-Be-eta1-random-{i}' for i in range(1, 7)]
stan_runs = ['const-Be-eta1-signaling-similarity']
list_ranges = [(1, 500)]
# list_ranges = [(1, 250)]
# list_ranges = [(1, 100)]
# list_ranges = [(1, 100), (1, 100), (1, 100), (1, 100), (1, 100)]
# list_ranges = [(1, 372)]
# list_ranges = [(1, 571), (1, 372), (1, 359), (1, 341), (1, 335), (1, 370)]

cluster_method = 'k_means'
num_clusters = 3
cluster_key = f'{cluster_method}_{num_clusters}'

# get cell list
with open('stan_run_meta.json', 'r') as f:
    stan_run_meta = json.load(f)
session_list = []
for run, lr in zip(stan_runs, list_ranges):
    cell_list_path = os.path.join('cell_lists',
                                  stan_run_meta[run]['cell_list'])
    run_cell_list = pd.read_csv(cell_list_path, sep='\t')
    cell_list = run_cell_list.iloc[lr[0]:lr[1] + 1, :]
    run_root = os.path.join('../../result', stan_run_meta[run]['output_dir'])
    session_list.extend([str(c) for c in cell_list['Cell']])

session_list_int = [int(s) for s in session_list]

# get calcium response
t0 = 200
t_downsample = 300
y_all, y0_all, ts = load_trajectories(t0, filter_type='moving_average',
    moving_average_window=20, downsample_offset=t_downsample)
y_sessions = y_all[session_list_int, :]

# load all samples
num_runs = len(stan_runs)

if num_runs == 1:
    output_root = stan_run_meta[stan_runs[0]]['output_dir']
else:
    output_root = stan_run_meta[stan_runs[0]]['output_dir'][:-2] + '-all'
output_root = os.path.join('../../result', output_root)
if not os.path.exists(output_root):
    os.mkdir(output_root)
output_dir = os.path.join(output_root, 'trajectory-clustering')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
sc.settings.figdir = output_dir

# change font settings
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['font.size'] = 12

# %%
# load expression data and preprocess
print('Loading gene expression...')
adata = sc.read_csv('vol_adjusted_genes.csv')
adata = adata[session_list, :]
adata.raw = adata
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.scale(adata)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
# run clustering on trajectories
kmeans = TimeSeriesKMeans(n_clusters=num_clusters, metric='euclidean',
                          random_state=0)
adata.obs[cluster_key] = pd.Series(kmeans.fit_predict(y_sessions),
                                   index=adata.obs_names, dtype='category')
if cluster_key == 'k_means_3':
    if 'const-Be-eta1' in stan_runs or 'const-Be-eta1-signaling-similarity' in stan_runs:
        adata.rename_categories(cluster_key, ['Low', 'High', 'Medium'])
    else:
        adata.rename_categories(cluster_key, ['C1', 'C2', 'C3'])
cluster_names = adata.obs[cluster_key].cat.categories
adata.uns[f'{cluster_key}_colors'] = [f'C{i + 3}' for i in range(num_clusters)]

# find marker genes of each cluster
sc.tl.rank_genes_groups(adata, cluster_key)

# %%
# plot trajectories
reordered_session_indices = np.argsort(adata.obs[cluster_key])

plt.figure(figsize=(4, 6), dpi=300)
_ = sns.heatmap(y_sessions[reordered_session_indices, :], xticklabels=False,
                yticklabels=np.sort(adata.obs[cluster_key]))
plt.xlabel('Time')
plt.ylabel('Ca2+ response')
plt.tight_layout()
figure_path = os.path.join(output_dir, f'{cluster_key}_trajectories.pdf')
plt.savefig(figure_path)
figure_path = os.path.join(output_dir, f'{cluster_key}_trajectories.png')
plt.savefig(figure_path)
plt.close()

# %%
# plot peaks of trajectories on PCA
adata.obs['peak'] = np.amax(y_sessions, axis=1)
with plt.rc_context({"figure.figsize": (4, 4), "figure.dpi": (300)}):
    sc.pl.pca(adata, color='peak', use_raw=False, show=False,
              save='_trajectory_peaks.pdf')

# %%
# plot distribution of trajectory peaks in each cluster
plt.figure(figsize=(4, 2), dpi=300)
for i, c in enumerate(cluster_names):
    peaks = np.array(adata.obs['peak'][adata.obs[cluster_key] == c])
    sns.kdeplot(data=peaks, fill=True, alpha=0.2, label=c,
                color=adata.uns[f'{cluster_key}_colors'][i])
plt.xlim((0, 10))
plt.xlabel(r'Peak Ca${}^{2+}$ response (AU)')
plt.yticks(ticks=[])
plt.legend()
# plt.title(r'Ca${}^{2+}$ response clustering')
plt.tight_layout()
figure_path = os.path.join(output_dir, f'{cluster_key}_traj_peaks.pdf')
plt.savefig(figure_path)
figure_path = os.path.join(output_dir, f'{cluster_key}_traj_peaks.png')
plt.savefig(figure_path)
plt.close('all')

# %%
# make ribbon plot for trajectories in each cluster
t_plot_max = 100
num_plot_points = np.sum(ts <= t_plot_max)
ts_plot = ts[:num_plot_points]

fig, axs = plt.subplots(nrows=num_clusters, ncols=1, sharex=True,
                        figsize=(3, num_clusters + 1), dpi=300)
for i, cluster in enumerate(cluster_names):
    cluster_cells = [int(c) for c in session_list
                     if adata.obs.loc[c, cluster_key] == cluster]
    y_cluster = y_all[cluster_cells, :]
    y_mean = np.mean(y_cluster, axis=0)
    y_std = np.std(y_cluster, axis=0, ddof=1)
    y_mean = y_mean[:num_plot_points]
    y_std = y_std[:num_plot_points]

    cluster_color = adata.uns[f'{cluster_key}_colors'][i]
    axs[i].plot(ts_plot, y_mean, color=cluster_color)
    axs[i].fill_between(ts_plot, y_mean - y_std, y_mean + y_std,
                        color=cluster_color, alpha=0.2)
    axs[i].set_xlabel('Time')
    axs[i].set_ylim(bottom=0, top=4)
    axs[i].set_title(cluster)

plt.tight_layout()
figure_path = os.path.join(output_dir, f'{cluster_key}_traj_ribbon.pdf')
plt.savefig(figure_path)
figure_path = os.path.join(output_dir, f'{cluster_key}_traj_ribbon.png')
plt.savefig(figure_path)
plt.close()

# %%
# get similarity matrix
soptsc_vars = scipy.io.loadmat(
        '../../result/SoptSC/SoptSC_feature_100/workspace.mat')
similarity_matrix = soptsc_vars['W']

plt.figure(figsize=(6, 6), dpi=300)
_ = sns.heatmap(
    similarity_matrix[np.ix_(reordered_session_indices, reordered_session_indices)],
    xticklabels=False, yticklabels=np.sort(adata.obs[cluster_key]))
plt.tight_layout()
figure_path = os.path.join(output_dir, f'{cluster_key}_traj_similarity.pdf')
plt.savefig(figure_path)
plt.close()

# %%
# plot gene expression after clustering
# sc.settings.figdir = output_dir
mpl.rcParams['font.size'] = 18

# plot all genes
axes = sc.pl.heatmap(adata, var_names=adata.var_names, groupby=cluster_key,
                     use_raw=False, figsize=(4, 3), show=False, save=False)
axes['groupby_ax'].set_yticklabels(axes['groupby_ax'].get_yticklabels(),
                                   fontdict={'verticalalignment': 'center'},
                                   rotation=90)
axes['heatmap_ax'].set_xlabel('Genes')
axes['groupby_ax'].set_ylabel('')
figure_basename = os.path.join(output_dir, f'{cluster_key}_genes_all')
plt.savefig(figure_basename + '.pdf')
plt.savefig(figure_basename + '.png')
plt.close()

# plot select genes
axes = sc.pl.heatmap(adata, var_names=['PPP1CC', 'CCDC47'], groupby=cluster_key,
                     use_raw=False, figsize=(4, 3), show=False, save=False)
axes['heatmap_ax'].set_xticklabels(axes['heatmap_ax'].get_xticklabels(),
                                   rotation=0)
axes['groupby_ax'].set_yticklabels(axes['groupby_ax'].get_yticklabels(),
                                   fontdict={'verticalalignment': 'center'},
                                   rotation=90)
plt.ylabel('')
figure_basename = os.path.join(output_dir, f'{cluster_key}_genes_select')
plt.savefig(figure_basename + '.pdf')
plt.savefig(figure_basename + '.png')
plt.close()

# %%
# make heatmap for marker genes
mpl.rcParams['font.size'] = 18
axes = sc.pl.rank_genes_groups_heatmap(adata, n_genes=5, use_raw=False,
                                       figsize=(5, 3), dendrogram=False,
                                       show_gene_labels=True, show=False,
                                       save=False)
axes['heatmap_ax'].set_xticklabels(axes['heatmap_ax'].get_xticklabels(),
                                   rotation=90)
axes['groupby_ax'].set_yticklabels(axes['groupby_ax'].get_yticklabels(),
                                   fontdict={'verticalalignment': 'center'},
                                   rotation=90)
axes['groupby_ax'].set_ylabel('')
for t in axes['gene_groups_ax'].texts:
    t.set(rotation=0)
figure_basename = os.path.join(output_dir, f'{cluster_key}_genes_marker')
plt.savefig(figure_basename + '.pdf', bbox_inches='tight')
plt.savefig(figure_basename + '.png', bbox_inches='tight')
plt.close()

# %%
# make matrix plot for marker genes
mpl.rcParams['font.size'] = 18
_ = sc.pl.rank_genes_groups_matrixplot(
    adata, n_genes=10, use_raw=False, dendrogram=False, var_group_rotation=0,
    show=False, save=False)
figure_basename = os.path.join(output_dir, f'{cluster_key}_genes_marker_matrix')
plt.savefig(figure_basename + '.pdf', bbox_inches='tight')
plt.savefig(figure_basename + '.png', bbox_inches='tight')
plt.close()

# %%
adata.write(
    os.path.join(output_dir, f'{cluster_key}_adata.h5ad'))
