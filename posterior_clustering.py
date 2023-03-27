# %%
import os
import os.path
import itertools
import json

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import ks_2samp
import scipy.io
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
import seaborn as sns
import scanpy as sc

from stan_helpers import StanMultiSessionAnalyzer, load_trajectories, \
    simulate_trajectory, get_mode_continuous_rv
from sample_distance import get_kl_nn, get_jensen_shannon, get_l2_divergence
import calcium_models

# %%
# initialize computation of distances between posterior samples
# stan_runs = ['3']
# stan_runs = ['const-Be-eta1']
# stan_runs = ['const-Be-eta1-signaling-similarity']
# stan_runs = ['const-Be-eta1-mixed-1']
# stan_runs = [f'const-Be-eta1-mixed-{i}' for i in range(5)]
# stan_runs = ['const-Be-eta1-random-1']
stan_runs = [f'const-Be-eta1-random-{i}' for i in range(1, 7)]
# list_ranges = [(1, 500)]
# list_ranges = [(1, 250)]
# list_ranges = [(1, 100), (1, 100), (1, 100), (1, 100), (1, 100)]
# list_ranges = [(1, 359)]
list_ranges = [(1, 571), (1, 372), (1, 359), (1, 341), (1, 335), (1, 370)]
log_transform_samples = False
scale_samples = False
max_num_clusters = 3
excluded_params = []
# excluded_params = ['Ke', 'eta2', 'k3']

# load metadata
with open('stan_run_meta.json', 'r') as f:
    stan_run_meta = json.load(f)

param_mask = stan_run_meta[stan_runs[0]]['param_mask']
param_names = [calcium_models.param_names[i + 1]
               for i, mask in enumerate(param_mask) if mask == "1"]
param_names = ['sigma'] + param_names

# get cell list
num_runs = len(stan_runs)
session_list = []
session_dirs = []
for run, lr in zip(stan_runs, list_ranges):
    run_root = os.path.join('../../result', stan_run_meta[run]['output_dir'])
    cell_list_path = os.path.join('cell_lists',
                                  stan_run_meta[run]['cell_list'])
    run_cell_list = pd.read_csv(cell_list_path, sep='\t')
    cell_list = run_cell_list.iloc[lr[0]:lr[1] + 1, :]
    session_list.extend([str(c) for c in cell_list['Cell']])
    session_dirs.extend([os.path.join(run_root, 'samples', f'cell-{c:04d}')
                         for c in cell_list['Cell']])

if num_runs == 1:
    output_root = stan_run_meta[stan_runs[0]]['output_dir']
else:
    output_root = stan_run_meta[stan_runs[0]]['output_dir'][:-2] + '-all'

output_root = os.path.join('../../result', output_root)
if not os.path.exists(output_root):
    os.mkdir(output_root)
output_dir = os.path.join(output_root, 'multi-sample-analysis')

# change font settings
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['font.size'] = 12

# load all samples
print('Loading samples...')
analyzer = StanMultiSessionAnalyzer(session_list, output_dir, session_dirs,
                                    param_names=param_names)
session_list = analyzer.session_list
session_list_int = [int(s) for s in session_list]
num_sessions = len(session_list)

excluded_params.append('sigma')
param_names = [pn for pn in param_names if pn not in excluded_params]
param_names_on_plot = [calcium_models.params_on_plot[p] for p in param_names]
num_params = len(param_names)

# log transform posteriors
if log_transform_samples:
    session_samples = [
        np.log1p(a.get_samples(excluded_params=excluded_params).to_numpy())
        for a in analyzer.session_analyzers
    ]
else:
    session_samples = [
        a.get_samples(excluded_params=excluded_params).to_numpy()
        for a in analyzer.session_analyzers
    ]

if scale_samples:
    sample_max = np.amax([np.amax(s, axis=0) for s in session_samples],
                         axis=0)
    sample_min = np.amin([np.amin(s, axis=0) for s in session_samples],
                         axis=0)

    for i in range(num_sessions):
        session_samples[i] = \
            (session_samples[i] - sample_min) / (sample_max - sample_min)

sample_means = np.empty((num_sessions, num_params))
for i, sample in enumerate(session_samples):
    sample_means[i, :] = np.mean(sample, axis=0)
sample_means_min = np.amin(sample_means, axis=0)
sample_means_max = np.amax(sample_means, axis=0)
sample_means = (sample_means - sample_means_min) / \
    (sample_means_max - sample_means_min)

sample_modes = np.empty((num_sessions, num_params))
for i, sample in enumerate(session_samples):
    for j, param in enumerate(param_names):
        sample_modes[i, j] = get_mode_continuous_rv(sample[:, j],
                                                    method='kde')
sample_modes_min = np.amin(sample_modes, axis=0)
sample_modes_max = np.amax(sample_modes, axis=0)
sample_modes = (sample_modes - sample_modes_min) / \
    (sample_modes_max - sample_modes_min)

# get calcium response
ode_variant = stan_run_meta[stan_runs[0]]['ode_variant']
calcium_ode = getattr(calcium_models, f'calcium_ode_{ode_variant}')
t0 = 200
t_downsample = 300
y_all, y0_all, ts = load_trajectories(t0, filter_type='moving_average',
    moving_average_window=20, downsample_offset=t_downsample)
y_sessions = y_all[session_list_int, :]

# %%
# get expression data
print('Loading gene expression...')
sc.settings.verbosity = 0
adata = sc.read_csv('../../data/vol_adjusted_genes.csv')
adata = adata[session_list, :]
adata.raw = adata
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.scale(adata)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# add parameter stats to adata
for i, param in enumerate(param_names):
    adata.obs[f'{param}_mean'] = sample_means[:, i]
    adata.obs[f'{param}_mode'] = sample_modes[:, i]

# set up folder for saving results
sample_cluster_root = os.path.join(output_root, 'posterior-clustering')
if log_transform_samples:
    sample_cluster_root += '-log-transformed'
if scale_samples:
    sample_cluster_root += '-scaled'
if len(excluded_params) > 1:
    sample_cluster_root += '_no'
    for pn in excluded_params:
        if pn != 'sigma':
            sample_cluster_root += f'_{pn}'
if not os.path.exists(sample_cluster_root):
    os.mkdir(sample_cluster_root)

sample_cluster_dir = os.path.join(sample_cluster_root,
                                  f'max-clusters-{max_num_clusters}')
if not os.path.exists(sample_cluster_dir):
    os.mkdir(sample_cluster_dir)

computed_cluster_keys = set()

# %%
# compute distances and save
sample_dists = {}
sample_dists['kl_yao'] = get_kl_nn(session_samples, verbose=True)
np.save(os.path.join(sample_cluster_root, 'kl_yao.npy'), sample_dists['kl_yao'])

# %%
for k in [2, 3, 5, 10]:
    kl_key =f'kl_{k}'
    sample_dists[kl_key] = get_kl_nn(session_samples, method='neighbor_any',
                                     k=k, verbose=True)
    np.save(os.path.join(sample_cluster_root, f'{kl_key}.npy'),
            sample_dists[kl_key])

    kl_key =f'kl_{k}_frac'
    sample_dists[kl_key] = get_kl_nn(
        session_samples, method='neighbor_fraction', k=k, verbose=True)
    np.save(os.path.join(sample_cluster_root, f'{kl_key}.npy'),
            sample_dists[kl_key])

# %%
for k, n in itertools.product([5, 10, 20], [20, 50, 100]):
    if k < n:
        l2_key = f'l2_{k}_{n}'
        sample_dists[l2_key] = get_l2_divergence(
            session_samples, k=k, subsample_size=n, verbose=True)

        np.save(os.path.join(sample_cluster_root, f'{l2_key}.npy'),
                sample_dists[l2_key])

        for i, j in itertools.combinations(range(num_sessions), 2):
            sample_dists[l2_key][i, j] = sample_dists[l2_key][j, i] \
                = min(sample_dists[l2_key][i, j], sample_dists[l2_key][j, i])

            if (sample_dists[l2_key][i, j] < 0):
                sample_dists[l2_key][i, j] = sample_dists[l2_key][j, i] = 0

# %%
sample_dists['js'] = get_jensen_shannon(session_samples)
np.save(os.path.join(sample_cluster_root, 'js.npy'), sample_dists['js'])

# %%
# load saved distance matrix if necessary
sample_dists = {}

sample_dists['kl_yao'] = np.load(
    os.path.join(sample_cluster_root, 'kl_yao.npy'))
sample_dists['kl_yao_1'] = np.load(
    os.path.join(sample_cluster_root, 'kl_yao_1.npy'))

for k in [2, 3, 5, 10]:
    kl_key =f'kl_{k}'
    sample_dists[kl_key] = np.load(os.path.join(sample_cluster_root,
                                                f'{kl_key}.npy'))

    kl_key =f'kl_{k}_frac'
    sample_dists[kl_key] = np.load(os.path.join(sample_cluster_root,
                                                f'{kl_key}.npy'))

# %%
for k, n in itertools.product([5, 10, 20], [20, 50, 100]):
    if k < n:
        l2_key = f'l2_{k}_{n}'
        sample_dists[l2_key] = np.load(os.path.join(sample_cluster_root,
                                                    f'{l2_key}.npy'))

        for i, j in itertools.combinations(range(num_sessions), 2):
            sample_dists[l2_key][i, j] = sample_dists[l2_key][j, i] \
                = min(sample_dists[l2_key][i, j], sample_dists[l2_key][j, i])

            if (sample_dists[l2_key][i, j] < 0):
                sample_dists[l2_key][i, j] = sample_dists[l2_key][j, i] = 0

# %%
sample_dists['js'] = np.load(os.path.join(sample_cluster_root, 'js.npy'))
sample_dists['js_10000'] = np.load(
    os.path.join(sample_cluster_root, 'js_10000.npy'))

# %%
# plot parameter stats on PCA/UMAP
sc.settings.figdir = sample_cluster_root

with plt.rc_context({"figure.figsize": (4, 4), "figure.dpi": (300)}):
    for param in param_names:
        # plot parameter mean
        _ = sc.pl.pca(adata, color=f'{param}_mean', save=f'_{param}_mean.pdf',
                      show=False)
        plt.close('all')

        _ = sc.pl.umap(adata, color=f'{param}_mean', save=f'_{param}_mean.pdf',
                       show=False)
        plt.close('all')

        # plo parametert mode
        _ = sc.pl.pca(adata, color=f'{param}_mode', save=f'_{param}_mode.pdf',
                      show=False)
        plt.close('all')

        _ = sc.pl.umap(adata, color=f'{param}_mode', save=f'_{param}_mode.pdf',
                       show=False)
        plt.close('all')

sc.settings.figdir = sample_cluster_dir

# %%
def cluster_by_sample_distances(dist_metric, cluster_method, plot=False):
    '''Cluster cells according to disances between posterior samples'''
    # set directory for figures
    cluster_key = f'{dist_metric}_{cluster_method}'

    # cluster with a distance matrix (get linkage data)
    dist_mat_1d = squareform(sample_dists[dist_metric])
    Z = linkage(dist_mat_1d, method=cluster_method)

    # get cluster labels
    labels = fcluster(Z, max_num_clusters, criterion='maxclust')
    adata.obs[cluster_key] = labels
    adata.obs[cluster_key] = adata.obs[cluster_key].astype('category')
    computed_cluster_keys.add(cluster_key)

    if not plot:
        return

    result_dir = os.path.join(sample_cluster_dir,
                              cluster_key.replace('_', '-'))
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    sc.settings.figdir = result_dir

    cluster_session_orders = leaves_list(Z)
    cluster_colors = [f'C{l - 1}' for l in labels]

    # plot heatmap of distance matrix reordered by clustering result
    g = sns.clustermap(sample_dists[dist_metric], row_linkage=Z,
                      col_linkage=Z, xticklabels=False, yticklabels=False,
                      row_colors=cluster_colors, figsize=(6, 6))
    figure_path = os.path.join(result_dir, 'clustered_distances.pdf')
    g.savefig(figure_path, dpi=300)
    plt.close('all')

    with plt.rc_context({"figure.figsize": (4, 4), "figure.dpi": (300)}):
        # label cells on PCA
        _ = sc.pl.pca(adata, color=cluster_key, save='.pdf', show=False)
        plt.close('all')

        # label cells on UMAP
        _ = sc.pl.umap(adata, color=cluster_key, save='.pdf', show=False)
        plt.close('all')

    # plot trajectories reordered by clustering result
    g = sns.clustermap(y_sessions[cluster_session_orders, :],
                       row_linkage=Z, col_cluster=False, xticklabels=False,
                       yticklabels=False, row_colors=cluster_colors,
                       figsize=(4, 6), cbar_pos=None)
    g.ax_heatmap.set_xlabel('Ca2+ response')
    g.ax_heatmap.set_ylabel('Cells')
    plt.tight_layout()
    figure_path = os.path.join(result_dir, 'clustered_trajectories.pdf')
    plt.savefig(figure_path)
    plt.close('all')

    # plot gene expression reordered by clustering result
    # g = sc.pl.clustermap(adata, obs_keys=cluster_key, use_raw=False,
    #                      row_linkage=Z, col_cluster=False,
    #                      xticklabels=False, yticklabels=False,
    #                      save=f'_gene_expression.pdf', show=False)

    # plot posterior means reordered by clustering result
    plt.figure(figsize=(4, 6), dpi=300)
    g = sns.clustermap(sample_means, row_linkage=Z,
                       col_cluster=False, xticklabels=param_names,
                       yticklabels=False, row_colors=cluster_colors,
                       figsize=(4, 6))#, cbar_pos=None)
    g.ax_heatmap.set_ylabel('Cells')
    plt.tight_layout()
    figure_path = os.path.join(result_dir, 'clustered_posterior_means.pdf')
    g.savefig(figure_path, dpi=300)
    plt.close('all')

    # plot posterior modes reordered by clustering result
    plt.figure(figsize=(4, 6), dpi=300)
    g = sns.clustermap(sample_modes, row_linkage=Z,
                       col_cluster=False, xticklabels=param_names,
                       yticklabels=False, row_colors=cluster_colors,
                       figsize=(4, 6), cbar_pos=None)
    g.ax_heatmap.set_ylabel('Cells')
    plt.tight_layout()
    figure_path = os.path.join(result_dir, 'clustered_posterior_modes.pdf')
    g.savefig(figure_path, dpi=300)
    plt.close('all')

# %%
dist_metrics = list(sample_dists.keys())
cluster_methods = ['ward']

for metric, method in itertools.product(dist_metrics, cluster_methods):
    print(f'Clustering {metric} using {method} linkage...')
    cluster_by_sample_distances(metric, method, plot=False)

# %%
figure_title = 'Posterior clustering'
if 'const-Be-eta1' in stan_runs:
    figure_title += ', similar'
elif 'const-Be-eta1-random-1' in stan_runs:
    figure_title += ', random'
else:
    figure_title = None

def cluster_by_sample_stats(stat_type, cluster_method, plot=False):
    cluster_key = f'{stat_type}_{cluster_method}'

    if stat_type == 'mean':
        data = sample_means
    else:  # data_type == 'mode'
        data = sample_modes

    if cluster_method == 'kmeans':
        kmeans = KMeans(n_clusters=max_num_clusters, random_state=0)
        labels = kmeans.fit_predict(data)
    else:
        Z = linkage(data, method=cluster_method)
        labels = fcluster(Z, max_num_clusters, criterion='maxclust')

    adata.obs[cluster_key] = labels
    adata.obs[cluster_key] = adata.obs[cluster_key].astype('category')
    computed_cluster_keys.add(cluster_key)

    if not plot:
        return

    result_dir = os.path.join(sample_cluster_dir,
                              'posterior-' + cluster_key.replace('_', '-'))
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    sc.settings.figdir = result_dir

    if cluster_method == 'kmeans':
        reordered_session_indices = np.argsort(labels)
        reordered_labels = labels[reordered_session_indices]
        # reordered_sessions = [session_list_int[i]
        #                       for i in reordered_session_indices]

        # plot trajectories reordered by clustering result
        plt.figure(figsize=(4, 6), dpi=300)
        _ = sns.heatmap(y_sessions[reordered_session_indices, :],
                        xticklabels=False, yticklabels=reordered_labels)
        plt.xlabel('Time')
        plt.ylabel('Ca2+ response')
        plt.tight_layout()
        figure_path = os.path.join(result_dir, 'clustered_trajectories.pdf')
        plt.savefig(figure_path)
        plt.close('all')

        # plot posterior means reordered by clustering result
        plt.figure(figsize=(4, 6), dpi=300)
        g = sns.heatmap(sample_means[reordered_session_indices, :],
                        xticklabels=param_names_on_plot,
                        yticklabels=reordered_labels)
        plt.ylabel('Cells')
        plt.xticks(fontsize=10)
        plt.tight_layout()
        figure_path = os.path.join(result_dir, 'clustered_posterior_means.pdf')
        plt.savefig(figure_path)
        plt.close('all')

        # plot posterior modes reordered by clustering result
        plt.figure(figsize=(4, 6), dpi=300)
        _ = sns.heatmap(sample_modes[reordered_session_indices, :],
                        xticklabels=param_names_on_plot,
                        yticklabels=reordered_labels)
        plt.ylabel('Cells')
        plt.xticks(fontsize=10)
        plt.tight_layout()
        figure_path = os.path.join(result_dir, 'clustered_posterior_modes.pdf')
        plt.savefig(figure_path)
        plt.close('all')
    else:
        cluster_colors = [f'C{l - 1}' for l in labels]
        cluster_session_orders = leaves_list(Z)

        # plot trajectories reordered by clustering result
        g = sns.clustermap(y_sessions[cluster_session_orders, :],
                           row_linkage=Z, col_cluster=False, xticklabels=False,
                           yticklabels=False, row_colors=cluster_colors,
                           figsize=(4, 6), cbar_pos=None)
        g.ax_heatmap.set_xlabel('Cells')
        g.ax_heatmap.set_ylabel('Ca2+ response')
        plt.tight_layout()
        figure_path = os.path.join(result_dir, 'clustered_trajectories.pdf')
        plt.savefig(figure_path)
        plt.close('all')

        # plot posterior means reordered by clustering result
        g = sns.clustermap(sample_means, row_linkage=Z, col_cluster=False,
                           xticklabels=param_names_on_plot, yticklabels=False,
                           row_colors=cluster_colors, figsize=(4, 6),
                           cbar_kws={'orientation': 'horizontal',
                                     'label': 'Normalized posterior mean'},
                           cbar_pos=(0.45, 0.88, 0.3, 0.02))
        g.ax_heatmap.set_ylabel('Cells')
        g.cax.xaxis.set_label_position('top')
        plt.xticks(fontsize=10)
        plt.tight_layout()
        figure_path = os.path.join(result_dir, 'clustered_posterior_means.pdf')
        g.savefig(figure_path, dpi=300)
        plt.close('all')

        # plot posterior modes reordered by clustering result
        g = sns.clustermap(sample_modes, row_linkage=Z, col_cluster=False,
                           xticklabels=param_names_on_plot, yticklabels=False,
                           row_colors=cluster_colors, figsize=(4, 6),
                           cbar_pos=None)
        g.ax_heatmap.set_ylabel('Cells')
        plt.xticks(fontsize=10)
        plt.tight_layout()
        figure_path = os.path.join(result_dir, 'clustered_posterior_modes.pdf')
        g.savefig(figure_path, dpi=300)
        plt.close('all')

    with plt.rc_context({"figure.figsize": (4, 4), "figure.dpi": (300)}):
        # label cells on PCA
        _ = sc.pl.pca(adata, color=cluster_key, save='.pdf', show=False)
        plt.close('all')

        # label cells on UMAP
        _ = sc.pl.umap(adata, color=cluster_key, save='.pdf', show=False)
        plt.close('all')

# %%
# cluster by posterior means or posterior modes
cluster_methods = ['ward']#, 'kmeans']
stat_types = ['mean']#, 'mode']
mpl.rcParams['font.size'] = 12

for st, method in itertools.product(stat_types, cluster_methods):
    if method != 'kmeans':
        print(f'Clustering posterior {st} using {method} linkage...')
    else:
        print(f'Clustering posterior {st} using k-means...')

    cluster_by_sample_stats(st, method, plot=True)

    # rename cluster names for plotting
    if 'const-Be-eta1' in stan_runs and st == 'mean' and method == 'ward':
        cluster_key = 'mean_ward'
        cluster_names = ['Early', 'Low', 'Late-high']
        adata.rename_categories(cluster_key, cluster_names)
        cluster_colors = ['C0', 'C1', 'C2']
    elif 'const-Be-eta1-random-1' in stan_runs and st == 'mean' \
            and method == 'ward':
        cluster_key = 'mean_ward'
        cluster_names = ['C1', 'C2', 'C3']
        adata.rename_categories(cluster_key, cluster_names)
        cluster_colors = ['C6', 'C7', 'C8']
    elif 'const-Be-eta1-signaling-similarity' in stan_runs and st == 'mean' \
            and method == 'ward':
        cluster_key = 'mean_ward'
        cluster_names = ['C1', 'C2', 'C3']
        adata.rename_categories(cluster_key, cluster_names)
        cluster_colors = ['C6', 'C7', 'C8']
    else:
        cluster_colors = [f'C{i}' for i in range(len(cluster_names))]

# %%
# find differential genes in each cluster
marker_gene_tests = ['t-test', 'wilcoxon', 't-test_overestim_var']
num_top_genes = 100
marker_gene_max_pval = 1.01
mpl.rcParams['font.size'] = 18

for cluster_key, test in itertools.product(computed_cluster_keys,
                                           marker_gene_tests):
    print(f'Finding marker genes for {cluster_key} using {test}...')
    metric, method = cluster_key.split('_')
    result_dir = f'{metric}-{method}'
    if metric in ('mean', 'mode'):
        result_dir = 'posterior-' + result_dir
    result_dir = os.path.join(sample_cluster_dir, result_dir)
    sc.settings.figdir = result_dir

    # get marker genes
    marker_gene_key = f'{cluster_key}_{test}'
    sc.tl.rank_genes_groups(adata, cluster_key, n_genes=num_top_genes,
                            method=test, key_added=marker_gene_key)

    marker_gene_table = sc.get.rank_genes_groups_df(
        adata, None, key=marker_gene_key, pval_cutoff=marker_gene_max_pval)
    marker_gene_table_path = os.path.join(result_dir,
                                          f'{test}_marker_genes.csv')
    marker_gene_table.to_csv(marker_gene_table_path)

    # plot marker genes
    # sc.pl.rank_genes_groups(
    #     adata, n_genes=10, key=marker_gene_key, sharey=False,
    #     save=f'_{test}.pdf', show=False)

    g = sc.pl.rank_genes_groups_heatmap(
        adata, n_genes=min(10, 50 // max_num_clusters), groupby=cluster_key,
        key=marker_gene_key, dendrogram=False, use_raw=False, show=False,
        save=f'_{test}_marker_genes.pdf')
    plt.close('all')

    g = sc.pl.rank_genes_groups_dotplot(
        adata, n_genes=min(10, 50 // max_num_clusters), groupby=cluster_key,
        key=marker_gene_key, dendrogram=False, use_raw=False, show=False,
        save=f'_{test}_marker_genes.pdf')
    plt.close('all')

    g = sc.pl.rank_genes_groups_matrixplot(
        adata, n_genes=min(10, 50 // max_num_clusters), groupby=cluster_key,
        key=marker_gene_key, dendrogram=False, var_group_rotation=0,
        use_raw=False, show=False, save=f'_{test}_marker_genes.pdf')
    plt.close('all')

    g = sc.pl.rank_genes_groups_violin(
        adata, n_genes=10, key=marker_gene_key, use_raw=False, show=False,
        save=f'_{test}.pdf')
    plt.close('all')

# %%
# get GO terms for marker genes in clusters
from goatools.base import download_go_basic_obo
from goatools.obo_parser import GODag
from goatools.base import download_ncbi_associations
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.test_data.genes_NCBI_9606_ProteinCoding import GENEID2NT \
    as GeneID2nt
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS

# initialize GO analysis
cluster_key = 'posterior_mean_ward'
marker_gene_test = 't-test'
filter_col = 'pvals'
# filter_col = 'pvals_adj'
marker_gene_max_pval = 0.05
go_max_pval = 0.05

obo_path = download_go_basic_obo()
obo_dag = GODag(obo_path)
gene2go_path = download_ncbi_associations()
gene_annoation = Gene2GoReader(gene2go_path, taxids=[9606])
ns2assoc = gene_annoation.get_ns2assc()

# create reverse mapping from gene symbols to gene IDs
symbol2id = {}
for gene_id, nt in GeneID2nt.items():
    gene_symbol = nt.Symbol.upper()
    symbol2id[gene_symbol] = gene_id

    for alias in nt.Aliases.split(', '):
        symbol2id[alias.upper()] = gene_id

# get gene IDs for all genes in the dataset
pop_ids = set()
for gene in adata.var_names:
    pop_ids.add(symbol2id[gene])

go_study = GOEnrichmentStudyNS(pop_ids, ns2assoc, obo_dag,
                               propagate_counts = False, alpha = go_max_pval,
                               methods = ['fdr_bh'])
result_dir = os.path.join(sample_cluster_dir, cluster_key.replace('_', '-'))
marker_gene_table_path = os.path.join(
    result_dir, f'{marker_gene_test}_marker_genes.csv')
marker_gene_table = pd.read_csv(marker_gene_table_path, index_col=0)
go_tables = {}

for cluster in np.unique(marker_gene_table['group']):
    # get gene ids from gene symbols
    gene_symbols = marker_gene_table['names'][
        (marker_gene_table['group'] == cluster)
        & (marker_gene_table[filter_col] < marker_gene_max_pval)].tolist()
    gene_ids = [symbol2id[s] for s in gene_symbols]
    go_tables[cluster] = pd.DataFrame(
        columns=['go_id', 'term', 'p_val_adj', 'gene_count', 'genes'])
    go_tables[cluster] = go_tables[cluster].astype({'gene_count': int})

    # run GO study
    if len(gene_ids) > 0:
        go_records = go_study.run_study(gene_ids)

        # collect all significant GO terms
        for gr in go_records:
            if gr.p_fdr_bh < go_max_pval:
                go_genes = [GeneID2nt[i].Symbol.upper() for i in gr.study_items]
                row = {'go_id': gr.goterm.id, 'term': gr.name,
                    'p_val_adj': gr.p_fdr_bh, 'gene_count': gr.study_count,
                    'genes': '.'.join(go_genes)}
                go_tables[cluster] = go_tables[cluster].append(
                    row, ignore_index=True)

        # sort GO terms by adjusted p-value
        go_tables[cluster].sort_values('p_val_adj', ascending=False)

    # export GO table
    go_table_path = os.path.join(
        result_dir, f'go_terms_{marker_gene_test}_{cluster}_{filter_col}.csv')
    go_tables[cluster].to_csv(go_table_path)

# %%
# compute trajectory statistics in each cluster
cluster_traj_stats = ['PeakMean', 'PeakStd', 'PeakTimeMean', 'PeakTimeStd',
                      'SteadyMean', 'SteadyStd']
cluster_traj_stat_table = {}
num_steady_pts = ts.size // 5
mpl.rcParams['font.size'] = 12

for cluster_key in computed_cluster_keys:
    metric, method = cluster_key.split('_')
    result_dir = f'{metric}-{method}'
    if metric in ('mean', 'mode'):
        result_dir = 'posterior-' + result_dir
    result_dir = os.path.join(sample_cluster_dir, result_dir)
    cluster_names = adata.obs[cluster_key].cat.categories
    cluster_traj_stat_table[cluster_key] = pd.DataFrame(
        columns=cluster_traj_stats, index=cluster_names)
    cluster_traj_features = {'peaks': [], 'peak_times': [],
                             'steady_states': []}

    for cluster in cluster_names:
        cluster_cells = [int(c) for c in session_list
                         if adata.obs.loc[c, cluster_key] == cluster]
        y_cluster = y_all[cluster_cells, :]
        row = {}

        # compute peak stats
        peaks = np.amax(y_cluster, axis=1)
        row['PeakMean'] = np.mean(peaks)
        row['PeakStd'] = np.std(peaks)
        cluster_traj_features['peaks'].append(peaks)

        # compute peak time stats
        peak_times = np.argmax(y_cluster, axis=1)
        row['PeakTimeMean'] = np.mean(peak_times)
        row['PeakTimeStd'] = np.std(peak_times)
        cluster_traj_features['peak_times'].append(peak_times)

        # compute steady state stats
        steady_states = np.mean(y_cluster[:, -num_steady_pts:], axis=1)
        row['SteadyMean'] = np.mean(steady_states)
        row['SteadyStd'] = np.std(steady_states)
        cluster_traj_features['steady_states'].append(steady_states)

        cluster_traj_stat_table[cluster_key].loc[cluster, :] = row

    table_path = os.path.join(result_dir, 'traj_stats.csv')
    cluster_traj_stat_table[cluster_key].to_csv(table_path)

    # plot histogram of each feature
    for feature, values in cluster_traj_features.items():
        plt.figure(figsize=(4, 2), dpi=300)
        if feature == 'peaks':
            # for peak distribution, plot KDE instead
            for cn, v, cc in zip(cluster_names, values, cluster_colors):
                sns.kdeplot(data=v, color=cc, fill=True, alpha=0.2, label=cn)
            plt.xlim((0, 10))
            plt.yticks(ticks=[])
        else:
            plt.hist(values, bins=10, density=True, label=cluster_names)
        plt.legend()
        if figure_title:
            plt.title(figure_title)

        plt.tight_layout()

        figure_path = os.path.join(result_dir, f'traj_{feature}.pdf')
        plt.savefig(figure_path)
        plt.close()

    # test peak time:
    peak_time_ks_table = pd.DataFrame(
        columns=['Cluster A', 'Cluster B', 'KS stat', 'p-value'])
    for i, j in itertools.combinations(range(len(cluster_names)), 2):
        ks, p_val = ks_2samp(cluster_traj_features['peak_times'][i],
                             cluster_traj_features['peak_times'][j],
                             alternative='less')
        row = {'Cluster A': cluster_names[i], 'Cluster B': cluster_names[j],
               'KS stat': ks, 'p-value': p_val}
        peak_time_ks_table = peak_time_ks_table.append(row, ignore_index=True)

        ks, p_val = ks_2samp(cluster_traj_features['peak_times'][j],
                             cluster_traj_features['peak_times'][i],
                             alternative='less')
        row = {'Cluster A': cluster_names[j], 'Cluster B': cluster_names[i],
               'KS stat': ks, 'p-value': p_val}
        peak_time_ks_table = peak_time_ks_table.append(row, ignore_index=True)

    table_path = os.path.join(result_dir, 'traj_peak_time_ks.csv')
    peak_time_ks_table.to_csv(table_path)

# %%
# make ribbon plot for trajectories in each cluster
t_plot_max = 100
num_plot_points = np.sum(ts <= t_plot_max)
ts_plot = ts[:num_plot_points]

for cluster_key in computed_cluster_keys:
    metric, method = cluster_key.split('_')
    result_dir = f'{metric}-{method}'
    if metric in ('mean', 'mode'):
        result_dir = 'posterior-' + result_dir
    result_dir = os.path.join(sample_cluster_dir, result_dir)
    cluster_names = adata.obs[cluster_key].cat.categories
    num_clusters = len(cluster_names)

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

        axs[i].plot(ts_plot, y_mean, color=f'C{i}')
        axs[i].fill_between(ts_plot, y_mean - y_std, y_mean + y_std,
                            color=f'C{i}', alpha=0.2)

        # add a vertical line at peak time
        peak_time = ts_plot[np.argmax(y_mean)]
        axs[i].plot(np.full(2, peak_time), np.array([0, 4]), 'k--')

        if i == num_clusters - 1:
            axs[i].set_xlabel('Time')
        axs[i].set_ylim(bottom=0, top=4)
        axs[i].set_title(f'Cluster {cluster}')

    plt.tight_layout()
    figure_path = os.path.join(result_dir, 'traj_ribbon.pdf')
    plt.savefig(figure_path)
    plt.close()

# %%
# plot trajectories for PLC, IP3, h for randomly selected cells
import tqdm

random_seed = 0
bit_generator = np.random.MT19937(random_seed)
rng = np.random.default_rng(bit_generator)
traj_sim_step_size = 50
num_cluster_sample_cells = 20
num_chain_samples = analyzer.session_analyzers[0].samples[0].shape[0]
num_chain_trajs = num_chain_samples // traj_sim_step_size
num_cell_trajs = num_chain_trajs * analyzer.num_chains
num_cluster_trajs = num_cluster_sample_cells * num_cell_trajs

for cluster_key in computed_cluster_keys:
    metric, method = cluster_key.split('_')
    result_dir = f'{metric}-{method}'
    if metric in ('mean', 'mode'):
        result_dir = 'posterior-' + result_dir
    result_dir = os.path.join(sample_cluster_dir, result_dir)
    cluster_names = adata.obs[cluster_key].cat.categories
    num_clusters = len(cluster_names)

    plc_trajs = {}
    ip3_trajs = {}
    h_trajs = {}

    for i, cluster in enumerate(cluster_names):
        print(f'Simulating for cluster {cluster}...', flush=True)
        cluster_cell_indices = [idx for idx, c in enumerate(session_list)
                                if adata.obs.loc[c, cluster_key] == cluster]
        chosen_cell_indices = rng.choice(
            cluster_cell_indices, size=num_cluster_sample_cells, replace=False)
        plc_trajs[cluster] = np.empty((num_cluster_trajs, ts.size))
        ip3_trajs[cluster] = np.empty((num_cluster_trajs, ts.size))
        h_trajs[cluster] = np.empty((num_cluster_trajs, ts.size))
        traj_idx = 0

        for idx in tqdm.tqdm(chosen_cell_indices):
            y0 = np.array([0, 0, 0.7, y0_all[idx]])
            y_sim = analyzer.session_analyzers[idx].simulate_chains(
                calcium_ode, 0, ts, y0, subsample_step_size=traj_sim_step_size,
                plot=False, verbose=False)
            for chain in range(analyzer.num_chains):
                plc_trajs[cluster][traj_idx:traj_idx+num_chain_trajs, :] \
                    = y_sim[chain][:, :, 0]
                ip3_trajs[cluster][traj_idx:traj_idx+num_chain_trajs, :] \
                    = y_sim[chain][:, :, 1]
                h_trajs[cluster][traj_idx:traj_idx+num_chain_trajs, :] \
                    = y_sim[chain][:, :, 2]

                traj_idx += num_chain_trajs

# %%
# plot samples of PLC, IP3, h
jet_cmap = plt.get_cmap('jet')
traj_colors = jet_cmap(np.linspace(0.0, 1.0, num_cluster_sample_cells))

for cluster_key in computed_cluster_keys:
    fig, axs = plt.subplots(nrows=3, ncols=num_clusters, sharex=True,
                            sharey='row', figsize=(2 * num_clusters, 4),
                            dpi=300)

    for i, cluster in enumerate(cluster_names):
        for j in range(num_cluster_sample_cells):
            traj_start_idx = j * num_cell_trajs
            traj_stop_idx = (j + 1) * num_cell_trajs
            axs[0][i].plot(
                ts, plc_trajs[cluster][traj_start_idx:traj_stop_idx, :].T,
                color=traj_colors[j], alpha=0.2)
            axs[0][i].set_title(cluster)
            axs[0][i].set_xticks([])
            axs[0][i].set_ylim(bottom=0, top=1)
            if i == 0:
                axs[0][i].set_ylabel("PLC")

            axs[1][i].plot(
                ts, ip3_trajs[cluster][traj_start_idx:traj_stop_idx, :].T,
                color=traj_colors[j], alpha=0.2)
            axs[1][i].set_xticks([])
            axs[1][i].set_ylim(bottom=0, top=20)
            if i == 0:
                axs[1][i].set_ylabel("IP3")

            axs[2][i].plot(
                ts, h_trajs[cluster][traj_start_idx:traj_stop_idx, :].T,
                color=traj_colors[j], alpha=0.2)
            axs[2][i].set_xticks([])
            axs[2][i].set_xlabel("Time")
            axs[2][i].set_ylim(bottom=0, top=1)
            if i == 0:
                axs[2][i].set_ylabel("h")

    fig.tight_layout()
    figure_path = os.path.join(result_dir, 'trajs_other_vars.png')
    fig.savefig(figure_path)
    plt.close('all')

for cluster_key in computed_cluster_keys:
    fig, axs = plt.subplots(nrows=3, ncols=num_clusters, sharex=True,
                            sharey='row', figsize=(2 * num_clusters, 4),
                            dpi=300)

    for i, cluster in enumerate(cluster_names):
        plc_mean = np.mean(plc_trajs[cluster], axis=0)
        plc_std = np.std(plc_trajs[cluster], axis=0, ddof=1)
        axs[0][i].plot(ts, plc_mean, color=f'C{i}')
        axs[0][i].fill_between(ts, plc_mean - plc_std, plc_mean + plc_std,
                               color=f'C{i}', alpha=0.2)
        axs[0][i].set_title(cluster)
        axs[0][i].set_xticks([])
        if i == 0:
            axs[0][i].set_ylabel("PLC")

        ip3_mean = np.mean(ip3_trajs[cluster], axis=0)
        ip3_std = np.std(ip3_trajs[cluster], axis=0, ddof=1)
        axs[1][i].plot(ts, ip3_mean, color=f'C{i}')
        axs[1][i].fill_between(ts, ip3_mean - ip3_std, ip3_mean + plc_std,
                               color=f'C{i}', alpha=0.2)
        axs[1][i].set_xticks([])
        if i == 0:
            axs[1][i].set_ylabel("IP3")

        h_mean = np.mean(h_trajs[cluster], axis=0)
        h_std = np.std(h_trajs[cluster], axis=0, ddof=1)
        axs[2][i].plot(ts, h_mean, color=f'C{i}')
        axs[2][i].fill_between(ts, h_mean - h_std, h_mean + h_std,
                               color=f'C{i}', alpha=0.2)
        axs[2][i].set_xticks([])
        axs[2][i].set_xlabel("Time")
        if i == 0:
            axs[2][i].set_ylabel("h")

    fig.tight_layout()
    figure_path = os.path.join(result_dir, 'trajs_other_vars_ribbon.pdf')
    fig.savefig(figure_path)
    plt.close('all')

# %%
# compare with Leiden clustering on genes
leiden_data_path = os.path.join(output_root, 'gene-clustering',
                                'leiden_0.50_adata.h5ad')
adata_leiden = sc.read_h5ad(leiden_data_path)
rand_scores = {}

for cluster_key in computed_cluster_keys:
    rand_scores[cluster_key] = adjusted_rand_score(
        adata_leiden.obs['leiden_0.50'], adata.obs[cluster_key])

    print(rand_scores[cluster_key])

# %%
# sample distance vs similarity
soptsc_vars = scipy.io.loadmat(
    '../../result/SoptSC/SoptSC_feature_100/workspace.mat')
similarity_mat = soptsc_vars['W'][np.ix_(session_list_int, session_list_int)]
similary_mat_1d = []
sampled_dists_1d = {d: [] for d in sample_dists}
for i in range(num_sessions):
    for j in range(i, num_sessions):
        similary_mat_1d.append(similarity_mat[i, j])

        for metric in sample_dists:
            sampled_dists_1d[metric].append(sample_dists[metric][i, j])

for metric in sample_dists:
    plt.figure(figsize=(6, 6), dpi=300)
    plt.scatter(similary_mat_1d, sampled_dists_1d[metric], alpha=0.3)
    plt.xlabel('Similarity')
    plt.ylabel(metric)
    plt.tight_layout()
    figure_path = os.path.join(sample_cluster_root,
                               f'{metric}_vs_similarity.pdf')
    plt.savefig(figure_path)
    plt.close()

    plt.figure(figsize=(6, 4), dpi=300)
    plt.hist(sampled_dists_1d[metric], bins=50)
    plt.xlabel(metric)
    plt.ylabel('Number of cell pairs')
    plt.tight_layout()
    figure_path = os.path.join(sample_cluster_root, f'{metric}_hist.pdf')
    plt.savefig(figure_path)
    plt.close()

# %%
# plot cell-cell similarity matrix after clustering
soptsc_vars = scipy.io.loadmat(
    '../../result/SoptSC/SoptSC_feature_100/workspace.mat')
similarity_mat = soptsc_vars['W'][np.ix_(session_list_int, session_list_int)]
similarity_mat = np.ceil(similarity_mat)

for cluster_key in computed_cluster_keys:
    cell_order = adata.obs[cluster_key].argsort(kind='mergesort').to_numpy()
    cluster_labels = adata.obs[cluster_key].to_numpy()[cell_order]
    cluster_names = adata.obs[cluster_key].cat.categories
    axis_ticks = np.cumsum(
        [0] + [np.sum(cluster_labels == cn) for cn in cluster_names])
    axis_tick_labels = [''] * len(axis_ticks)

    plt.figure(figsize=(4, 4), dpi=300)
    cluster_similarity_mat = similarity_mat[np.ix_(cell_order, cell_order)]
    for i, j in itertools.combinations(range(num_sessions), 2):
        cluster_similarity_mat[i, j] = 0.5
    plt.imshow(cluster_similarity_mat, cmap=plt.get_cmap('binary'))
    # plt.xticks(ticks=np.arange(len(cluster_labels)), labels=cluster_labels,
    #            rotation=90)
    plt.xticks(ticks=axis_ticks, labels=axis_tick_labels)
    # plt.yticks(ticks=np.arange(len(cluster_labels)), labels=cluster_labels)
    plt.yticks(ticks=axis_ticks, labels=axis_tick_labels)
    plt.tight_layout()

    metric, method = cluster_key.split('_')
    result_dir = f'{metric}-{method}'
    if metric in ('mean', 'mode'):
        result_dir = 'posterior-' + result_dir
    result_dir = os.path.join(sample_cluster_dir, result_dir)
    figure_path = os.path.join(result_dir, 'clustered_similarity.pdf')
    plt.savefig(figure_path)

    plt.close()

# %%
# cluster trajectories
traj_cluster_dir = os.path.join(output_root, 'trajectory-clustering')
if not os.path.exists(traj_cluster_dir):
    os.mkdir(traj_cluster_dir)

# hierarchical
cluster_methods = ['single', 'complete', 'average', 'centroid', 'median',
                   'ward']

for method in cluster_methods:
    print(f'Clustering trajectories using {method} linkage...')
    traj_linkage = linkage(y_sessions, method=method)
    traj_clusters = fcluster(traj_linkage, max_num_clusters,
                             criterion='maxclust')
    traj_cluster_colors = [f'C{cluster - 1}' for cluster in traj_clusters]
    reordered_sessions = leaves_list(traj_linkage)

    # plot trajectories reordered by clustering result
    g = sns.clustermap(y_sessions[reordered_sessions, :],
                       row_linkage=traj_linkage, col_cluster=False,
                       xticklabels=False, yticklabels=False,
                       row_colors=traj_cluster_colors, figsize=(4, 6))
    g.ax_heatmap.set_xlabel('Cells')
    g.ax_heatmap.set_ylabel('Ca2+ response')
    plt.tight_layout()
    figure_path = os.path.join(
        traj_cluster_dir, f'trajectories_{method}_{max_num_clusters}.pdf')
    plt.savefig(figure_path)
    plt.close('all')

# k-means
print('Clustering trajectories using k-means...')
kmeans = KMeans(n_clusters=max_num_clusters, random_state=0)
traj_labels = kmeans.fit_predict(y_sessions)
traj_cluster_cell_orders = np.argsort(traj_labels)

plt.figure(figsize=(4, 6), dpi=300)
_ = plt.imshow(y_sessions[traj_cluster_cell_orders, :])
figure_path = os.path.join(
    traj_cluster_dir, f'trajectories_kmeans_{max_num_clusters}.pdf')
plt.savefig(figure_path)
plt.close()

# %%
# compare parameters between samples
num_samples = 5000
random_seed = 0
bit_generator = np.random.MT19937(random_seed)
rng = np.random.default_rng(bit_generator)

for cluster_key in computed_cluster_keys:
    cluster_names = adata.obs[cluster_key].cat.categories
    cluster_samples = {}
    cluster_sample_stats = pd.DataFrame(index=cluster_names)
    metric, method = cluster_key.split('_')
    result_dir = f'{metric}-{method}'
    if metric in ('mean', 'mode'):
        result_dir = 'posterior-' + result_dir
    result_dir = os.path.join(sample_cluster_dir, result_dir)

    # sample from posterior
    for cluster in cluster_names:
        print(f'Sampling from cluster {cluster}...')
        cluster_cells = np.argwhere(
            (adata.obs[cluster_key] == cluster).to_numpy())
        cluster_samples[cluster] = pd.DataFrame(index=range(num_samples),
                                                columns=param_names)

        for i in range(num_samples):
            cell_idx = rng.choice(cluster_cells, size=1).item()
            cell_samples = analyzer.session_analyzers[cell_idx].get_samples(
                    excluded_params=excluded_params)
            cluster_samples[cluster].loc[i, :] = \
                cell_samples.sample(1, random_state=bit_generator).values

        for param in param_names:
            cluster_sample_stats.loc[cluster, f'{param}_mean'] = \
                cluster_samples[cluster][param].mean()
            cluster_sample_stats.loc[cluster, f'{param}_std'] = \
                cluster_samples[cluster][param].std()

    output_path = os.path.join(result_dir, 'sample_stats.csv')
    cluster_sample_stats.to_csv(output_path)

    # compare samples between clusters by KS test
    ks_results = pd.DataFrame(
        columns=['Cluster_1', 'Cluster_2', 'Parameter', 'H_1', 'KS', 'p_val'])

    for param in param_names:
        for c1, c2 in itertools.combinations(cluster_names, 2):
            for h1 in ['two-sided', 'less', 'greater']:
                ks_stat, ks_pval = ks_2samp(
                    cluster_samples[c1][param], cluster_samples[c2][param],
                    alternative=h1)

                row = {'Cluster_1': c1, 'Cluster_2': c2, 'Parameter': param,
                       'H_1': h1, 'KS': ks_stat, 'p_val': ks_pval}
                ks_results.loc[len(ks_results)] = row

    output_path = os.path.join(result_dir, 'sample_ks_test.csv')
    ks_results.to_csv(output_path)

    # plot sampled parameters
    print(f'Plotting histogram of samples for {cluster_key}...')
    figure_path = os.path.join(result_dir, 'sample_hist.pdf')
    with PdfPages(figure_path) as pdf:
        for param in param_names:
            plt.figure(figsize=(11, 8.5), dpi=300)

            # get parameter range
            param_min = np.inf
            param_max = -np.inf
            for cluster in cluster_names:
                param_min = min(param_min,
                                cluster_samples[cluster][param].min())
                param_max = max(param_max,
                                cluster_samples[cluster][param].max())

            # make histogram for all clusters
            for cluster in cluster_names:
                plt.hist(cluster_samples[cluster][param], bins=50,
                         range=(param_min, param_max), label=cluster,
                         alpha=0.3)

            plt.title(param)
            plt.legend()
            pdf.savefig()
            plt.close()

    # make box plots for parameter in the same figure
    print(f'Plotting box plot of samples for {cluster_key}...')
    num_cols = 4
    num_rows = (num_params - 1) // num_cols + 1
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols,
                            figsize=(6, num_rows * 1.5))

    for param_idx, param in enumerate(param_names):
        row = param_idx % num_cols
        col = param_idx // num_cols

        boxplot_data = pd.DataFrame(
            {c: cluster_samples[c][param] for c in cluster_names})
        sns.boxplot(data=boxplot_data, orient='h', fliersize=1, linewidth=0.5,
                    ax=axs[col, row])
        axs[col, row].set_xticks([])
        axs[col, row].set_yticks([])
        axs[col, row].set_title(calcium_models.params_on_plot[param])

    fig.tight_layout()
    figure_path = os.path.join(result_dir, 'sample_box_grid.pdf')
    plt.savefig(figure_path)
    plt.close('all')

    # make a standalone legend for clusters in box plots
    plt.figure(figsize=(4, 0.8), dpi=300)
    plt.axis('off')
    legend_patches = [mpatches.Patch(color=cc)
                      for cc in sns.color_palette(
                          palette='deep', n_colors=len(cluster_names))]
    plt.legend(legend_patches, cluster_names, loc='upper center',
               ncol=len(cluster_names), frameon=False,
               bbox_to_anchor=(0.0, 0.0))
    figure_path = os.path.join(result_dir, 'sample_box_grid_legend.pdf')
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close('all')

    # make box plots for parameters on separate pages
    figure_path = os.path.join(result_dir, 'sample_box.pdf')
    with PdfPages(figure_path) as pdf:
        for param in param_names:
            plt.figure(figsize=(11, 8.5), dpi=300)
            boxplot_data = pd.DataFrame(
                {c: cluster_samples[c][param] for c in cluster_names})
            sns.boxplot(data=boxplot_data)
            plt.title(param)
            pdf.savefig()
            plt.close()

    # make violin plots for parameters on separate pages
    print(f'Plotting violin plot of samples for {cluster_key}...')
    figure_path = os.path.join(result_dir, 'sample_violin.pdf')
    with PdfPages(figure_path) as pdf:
        for param in param_names:
            plt.figure(figsize=(11, 8.5), dpi=300)
            violin_data = pd.DataFrame(
                {c: cluster_samples[c][param] for c in cluster_names})
            sns.violinplot(data=violin_data)
            plt.title(param)
            pdf.savefig()
            plt.close()

# %%
# plot simulated trajecotries from each cluster
subsample_size = 500
random_seed = 0
bit_generator = np.random.MT19937(random_seed)
rng = np.random.default_rng(bit_generator)

for cluster_key in computed_cluster_keys:
    cluster_names = adata.obs[cluster_key].cat.categories
    result_dir = os.path.join(sample_cluster_dir,
                              cluster_key.replace('_', '-'))

    for cluster in cluster_names:
        # sample cells in a cluster
        cluster_cell_idxs = np.argwhere(
            adata.obs[cluster_key].to_numpy() == cluster)
        cluster_cell_idxs = np.squeeze(cluster_cell_idxs)
        subsample_cells = rng.choice(cluster_cell_idxs, size=subsample_size,
                                     replace=True)
        posterior_subsamples = np.empty(
            (subsample_size, len(analyzer.param_names)))
        subsample_trajs = np.empty((ts.size, subsample_size))

        # simulate trajectories from random samples
        for i, cell_idx in enumerate(subsample_cells):
            cell_id = session_list_int[cell_idx]
            cell_samples = analyzer.session_analyzers[cell_idx].get_samples()
            theta = cell_samples.sample(n=1, random_state=bit_generator).values
            theta = np.squeeze(theta[:, 1:])
            y0 = np.array([0, 0, 0.7, y0_all[cell_id]])
            y_sim = simulate_trajectory(calcium_ode, theta, 0, ts, y0)
            subsample_trajs[:, i] = y_sim[:, 3]

        # plot simulated trajectories
        plt.figure(figsize=(6, 4), dpi=300)
        plt.plot(ts, subsample_trajs)
        plt.xlabel('Time')
        plt.ylabel('Ca2+ response')
        plt.title(f'Cluster {cluster}')
        plt.tight_layout()
        figure_path = os.path.join(
            result_dir, f'simulated_trajectories_cluster_{cluster}.pdf')
        plt.savefig(figure_path)
        plt.close()

# %%
