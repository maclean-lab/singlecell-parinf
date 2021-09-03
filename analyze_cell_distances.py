# %%
import os
import os.path
import itertools
import json

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import scanpy as sc

from stan_helpers import StanMultiSessionAnalyzer, load_trajectories, \
    get_kl_nn, get_jensen_shannon
import calcium_models

# %%
# initialize computation of distances between posterior samples
stan_run = 'const-Be-eta1'
first_cell_order = 1
last_cell_order = 500
log_normalize_samples = True
num_max_clusters = 4

# load metadata
with open('stan_run_meta.json', 'r') as f:
    stan_run_meta = json.load(f)

param_mask = stan_run_meta[stan_run]['param_mask']
param_names = [calcium_models.param_names[i + 1]
               for i, mask in enumerate(param_mask) if mask == "1"]
param_names = ['sigma'] + param_names

# get cell list
cell_list_path = os.path.join(
    'cell_lists', stan_run_meta[stan_run]['cell_list'])
full_cell_list = pd.read_csv(cell_list_path, sep='\t')
cell_list = full_cell_list.iloc[first_cell_order:last_cell_order + 1, :]

# load all samples
print('Loading samples...')
output_root = os.path.join(
    '../../result', stan_run_meta[stan_run]['output_dir'])
output_dir = \
    f'multi-sample-analysis-{first_cell_order:04d}-{last_cell_order:04d}'
session_list = [str(c) for c in cell_list['Cell']]
session_dirs = [f'samples/cell-{c:04d}' for c in cell_list['Cell']]
session_dirs = [os.path.join(output_root, sd) for sd in session_dirs]
analyzer = StanMultiSessionAnalyzer(session_list, output_dir, session_dirs,
                                    param_names=param_names)

# log normalize posteriors
if log_normalize_samples:
    session_samples = [np.log1p(a.get_samples().iloc[:, 1:].to_numpy())
                       for a in analyzer.session_analyzers]
else:
    session_samples = [a.get_samples().iloc[:, 1:].to_numpy()
                       for a in analyzer.session_analyzers]

sample_means = np.empty((len(session_list), len(param_names) - 1))
for i, sample in enumerate(session_samples):
    sample_means[i, :] = np.mean(sample, axis=0)
sample_means_min = np.amin(sample_means, axis=0)
sample_means_max = np.amax(sample_means, axis=0)
sample_means = (sample_means - sample_means_min) / \
    (sample_means_max - sample_means_min)

# get calcium response
t0 = 200
t_downsample = 300
y_all, y0_all, ts = load_trajectories(t0, filter_type='moving_average',
    moving_average_window=20, downsample_offset=t_downsample)

# get expression data
print('Loading gene expression...')
adata = sc.read_csv('../../data/vol_adjusted_genes.csv')
adata = adata[analyzer.session_list, :]
adata.raw = adata
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.scale(adata)

# set up folder for saving results
sample_dist_dir = os.path.join(output_root, 'sample-dists')
if log_normalize_samples:
    sample_dist_dir += '-log-normalized'
if not os.path.exists(sample_dist_dir):
    os.mkdir(sample_dist_dir)
sample_cluster_dir = os.path.join(sample_dist_dir,
                                  f'max_clusters_{num_max_clusters}')
if not os.path.exists(sample_cluster_dir):
    os.mkdir(sample_cluster_dir)
sc.settings.figdir = sample_cluster_dir
sc.settings.verbosity = 0

# containers for clustering results
sample_dists = {}
cluster_labels = {}

# %%
# compute distances and save
sample_dists['kl_yao'] = get_kl_nn(session_samples)
np.save(os.path.join(sample_dist_dir, 'kl.npy'), sample_dists['kl_yao'])
sample_dists['js'] = get_jensen_shannon(session_samples)
np.save(os.path.join(sample_dist_dir, 'js.npy'), sample_dists['js'])

# %%
# load saved distance matrix if necessary
sample_dists['kl_yao'] = np.load(os.path.join(sample_dist_dir, 'kl.npy'))
sample_dists['js'] = np.load(os.path.join(sample_dist_dir, 'js.npy'))
sample_dists['js_10000'] = np.load(
    os.path.join(sample_dist_dir, 'js_10000.npy'))

# %%
def cluster_by_sample_distances(dist_metric, cluster_method, plot=False):
    '''Cluster cells according to disances between posterior samples'''
    # cluster with a distance matrix (get linkage data)
    dist_mat_1d = squareform(sample_dists[dist_metric])
    Z = linkage(dist_mat_1d, method=cluster_method)

    # get cluster labels
    cluster_labels[(dist_metric, cluster_method)] = \
        fcluster(Z, num_max_clusters, criterion='maxclust').astype(str)
    reordered_cell_indices = leaves_list(Z)
    reordered_sessions = [int(session_list[i]) for i in reordered_cell_indices]

    cluster_key = f'{dist_metric}_{cluster_method}'
    adata.obs[cluster_key] = cluster_labels[(dist_metric, cluster_method)]
    adata.obs[cluster_key] = adata.obs[cluster_key].astype('category')

    if plot:
        # plot heatmap of distance matrix reordered by clustering result
        cluster_colors = [
            f'C{int(cluster) - 1}'
            for cluster in cluster_labels[(dist_metric, cluster_method)]]
        g = sns.clustermap(sample_dists[dist_metric], row_linkage=Z,
                           col_linkage=Z, xticklabels=False, yticklabels=False,
                           row_colors=cluster_colors, figsize=(6, 6))
        figure_path = os.path.join(
            sample_cluster_dir, f'{dist_metric}_{cluster_method}_distances.pdf')
        g.savefig(figure_path, dpi=300)
        plt.close('all')

        # plot trajectories reordered by clustering result
        g = sns.clustermap(y_all[reordered_sessions, :], row_linkage=Z,
                           col_cluster=False, xticklabels=False,
                           yticklabels=False, row_colors=cluster_colors,
                           figsize=(4, 6))
        g.ax_heatmap.set_xlabel('Cells')
        g.ax_heatmap.set_ylabel('Ca2+ response')
        plt.tight_layout()
        figure_path = os.path.join(
            sample_cluster_dir,
            f'{dist_metric}_{cluster_method}_trajectories.pdf')
        plt.savefig(figure_path)
        plt.close('all')

        # plot gene expression reordered by clustering result
        # g = sc.pl.clustermap(adata, obs_keys=cluster_key, use_raw=False,
        #                      row_linkage=Z, col_cluster=False,
        #                      xticklabels=False, yticklabels=False,
        #                      save=f'_{cluster_key}_gene_expression.pdf',
        #                      show=False)

        # plot posterior means reordered by clustering result
        plt.figure(figsize=(4, 6), dpi=300)
        g = sns.clustermap(sample_means, row_linkage=Z,
                           col_cluster=False, xticklabels=param_names[1:],
                           yticklabels=False, row_colors=cluster_colors,
                           figsize=(4, 6))
        g.ax_heatmap.set_ylabel('Cells')
        figure_path = os.path.join(
            sample_cluster_dir,
            f'{dist_metric}_{cluster_method}_posterior_mean.pdf')
        g.savefig(figure_path, dpi=300)
        plt.close('all')

    return Z

# %%
dist_metrics = list(sample_dists.keys())
linkage_mat = {}
cluster_methods = ['single', 'complete', 'average', 'centroid', 'median',
                   'ward']

for metric, method in itertools.product(dist_metrics, cluster_methods):
    print(f'Clustering {metric} using {method} linkage...')
    linkage_mat[(metric, method)] = \
        cluster_by_sample_distances(metric, method, plot=True)

# %%
# find differential genes in each cluster
cluster_methods = ['ward']
marker_gene_tests = ['t-test', 'wilcoxon', 't-test_overestim_var']
num_top_genes = 10
p_val_max = 1.1

for metric, method, test in itertools.product(dist_metrics, cluster_methods,
                                              marker_gene_tests):
    # get marker genes
    cluster_key = f'{metric}_{method}'
    marker_gene_key = f'{metric}_{method}_{test}'
    sc.tl.rank_genes_groups(adata, cluster_key, n_genes=num_top_genes,
                            method=test, key_added=marker_gene_key)
    marker_gene_table = sc.get.rank_genes_groups_df(
        adata, None, key=marker_gene_key, pval_cutoff=p_val_max)

    # plot marker genes
    cluster_colors = [f'C{int(cluster) - 1}'
                      for cluster in cluster_labels[(metric, method)]]
    marker_gene_symbols = []
    for i in marker_gene_table.index:
        gene = marker_gene_table.loc[i, 'names']
        cluster = marker_gene_table.loc[i, 'group']
        marker_gene_symbols.append(f'{gene} ({cluster})')

    # g = sc.pl.clustermap(adata[:, marker_gene_table['names']],
    #                      obs_keys=cluster_key, use_raw=False,
    #                      row_linkage=linkage_mat[(metric, method)],
    #                      col_cluster=False, xticklabels=marker_gene_symbols,
    #                      yticklabels=False, show=False,
    #                      save=f'_{marker_gene_key}_marker_genes.pdf')
    g = sc.pl.rank_genes_groups_heatmap(
        adata, n_genes=num_top_genes, groupby=cluster_key,
        key=marker_gene_key, dendrogram=False, use_raw=False, show=False,
        save=f'_{marker_gene_key}_marker_genes.pdf')

    # sc.pl.rank_genes_groups(
    #     adata, n_genes=10, key=marker_gene_key, sharey=False,
    #     save=f'_{metric}_{method}_{num_max_clusters}_{test}.pdf',
    #     show=False)

# %%
# sample distance vs similarity
soptsc_vars = scipy.io.loadmat(
        '../../result/SoptSC/SoptSC_feature_100/workspace.mat')
cell_ids = [int(session) for session in session_list]
similary_mat = soptsc_vars['W'][np.ix_(cell_ids, cell_ids)]
similary_mat_1d = []
sampled_dists_1d = {m: [] for m in sample_dists}
for i in range(len(session_list)):
    for j in range(i, len(session_list)):
        similary_mat_1d.append(similary_mat[i, j])

        for metric in sample_dists:
            sampled_dists_1d[metric].append(sample_dists[metric][i, j])

for metric in sample_dists:
    plt.figure(figsize=(6, 6), dpi=300)
    plt.scatter(similary_mat_1d, sampled_dists_1d[metric], alpha=0.3)
    plt.xlabel('Similarity')
    plt.ylabel(metric)
    plt.tight_layout()
    figure_path = os.path.join(sample_dist_dir, f'{metric}_vs_similarity.pdf')
    plt.savefig(figure_path)
    plt.close()

    plt.figure(figsize=(6, 4), dpi=300)
    plt.hist(sampled_dists_1d[metric], bins=50)
    plt.xlabel(metric)
    plt.ylabel('Number of cell pairs')
    plt.tight_layout()
    figure_path = os.path.join(sample_dist_dir, f'{metric}_hist.pdf')
    plt.savefig(figure_path)
    plt.close()

# %%
# compare parameters between samples
cluster_methods = ['ward']
num_samples = 1000
random_seed = 0
bit_generator = np.random.MT19937(random_seed)
rng = np.random.default_rng(bit_generator)

for metric, method in itertools.product(sample_dists, cluster_methods):
    cluster_names = np.unique(cluster_labels[metric, method])
    cluster_samples = {}

    # sample from posterior
    print(f'Sampling for distance metric {metric}...')
    for cluster in cluster_names:
        print(f'Sampling from cluster {cluster} of {len(cluster_names)}...')
        cluster_cells = np.argwhere(cluster_labels[metric, method] == cluster)
        cluster_samples[cluster] = pd.DataFrame(index=range(num_samples),
                                                columns=param_names)

        for i in range(num_samples):
            cell_idx = rng.choice(cluster_cells, size=1).item()
            cell_samples = analyzer.session_analyzers[cell_idx].get_samples()
            cluster_samples[cluster].loc[i, :] = cell_samples.sample(1).values

    # plot sampled parameters
    print(f'Plotting samples...')
    figure_path = os.path.join(sample_cluster_dir,
                               f'{metric}_{method}_sample_hist.pdf')
    with PdfPages(figure_path) as pdf:
        for param in param_names[1:]:
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

# %%
