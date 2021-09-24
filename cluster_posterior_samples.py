# %%
import os
import os.path
import itertools
import json

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform
import scipy.io
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import scanpy as sc

from stan_helpers import StanMultiSessionAnalyzer, load_trajectories, \
    simulate_trajectory, get_mode_continuous_rv, get_kl_nn, get_jensen_shannon
import calcium_models

# %%
# initialize computation of distances between posterior samples
# stan_run = 'const-Be-eta1'
stan_run = 'const-Be-eta1-random-1'
# stan_run = 'const-Be-eta1-mixed-all'
first_cell_order = 1
last_cell_order = 200
log_normalize_samples = True
max_num_clusters = 3
# excluded_params = []
excluded_params = ['Ke', 'eta2', 'k3']

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
session_list = analyzer.session_list
session_list_int = [int(s) for s in session_list]

excluded_params.append('sigma')
param_names = [pn for pn in param_names if pn not in excluded_params]

# log normalize posteriors
if log_normalize_samples:
    session_samples = [
        np.log1p(a.get_samples(excluded_params=excluded_params).to_numpy())
        for a in analyzer.session_analyzers
    ]
else:
    session_samples = [
        a.get_samples(excluded_params=excluded_params).to_numpy()
        for a in analyzer.session_analyzers
    ]

sample_means = np.empty((len(session_list), len(param_names)))
for i, sample in enumerate(session_samples):
    sample_means[i, :] = np.mean(sample, axis=0)
sample_means_min = np.amin(sample_means, axis=0)
sample_means_max = np.amax(sample_means, axis=0)
sample_means = (sample_means - sample_means_min) / \
    (sample_means_max - sample_means_min)

sample_modes = np.empty((len(session_list), len(param_names)))
for i, sample in enumerate(session_samples):
    for j, param in enumerate(param_names):
        sample_modes[i, j] = get_mode_continuous_rv(sample[:, j],
                                                    method='kde')
sample_modes_min = np.amin(sample_modes, axis=0)
sample_modes_max = np.amax(sample_modes, axis=0)
sample_modes = (sample_modes - sample_modes_min) / \
    (sample_modes_max - sample_modes_min)

# get calcium response
ode_variant = stan_run_meta[stan_run]['ode_variant']
calcium_ode = getattr(calcium_models, f'calcium_ode_{ode_variant}')
t0 = 200
t_downsample = 300
y_all, y0_all, ts = load_trajectories(t0, filter_type='moving_average',
    moving_average_window=20, downsample_offset=t_downsample)
y_sessions = y_all[session_list_int, :]

# get expression data
print('Loading gene expression...')
adata = sc.read_csv('../../data/vol_adjusted_genes.csv')
adata = adata[session_list, :]
adata.raw = adata
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.scale(adata)

# set up folder for saving results
sample_cluster_root = os.path.join(output_root, 'posterior-clustering')
if log_normalize_samples:
    sample_cluster_root += '-log-normalized'
if len(excluded_params) > 1:
    dir_suffix = '_no'
    for pn in excluded_params:
        if pn != 'sigma':
            dir_suffix += f'_{pn}'

    sample_cluster_root += dir_suffix
if not os.path.exists(sample_cluster_root):
    os.mkdir(sample_cluster_root)
sample_cluster_dir = os.path.join(sample_cluster_root,
                                  f'max_clusters_{max_num_clusters}')
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
np.save(os.path.join(sample_cluster_root, 'kl.npy'), sample_dists['kl_yao'])
sample_dists['js'] = get_jensen_shannon(session_samples)
np.save(os.path.join(sample_cluster_root, 'js.npy'), sample_dists['js'])

# %%
# load saved distance matrix if necessary
sample_dists['kl_yao'] = np.load(os.path.join(sample_cluster_root, 'kl.npy'))
sample_dists['js'] = np.load(os.path.join(sample_cluster_root, 'js.npy'))
sample_dists['js_10000'] = np.load(
    os.path.join(sample_cluster_root, 'js_10000.npy'))

# %%
def cluster_by_sample_distances(dist_metric, cluster_method, plot=False):
    '''Cluster cells according to disances between posterior samples'''
    # cluster with a distance matrix (get linkage data)
    dist_mat_1d = squareform(sample_dists[dist_metric])
    Z = linkage(dist_mat_1d, method=cluster_method)
    linkage_mat[(dist_metric, cluster_method)] = Z

    # get cluster labels
    labels = fcluster(Z, max_num_clusters, criterion='maxclust')
    cluster_labels[(dist_metric, cluster_method)] = labels

    cluster_key = f'{dist_metric}_{cluster_method}'
    adata.obs[cluster_key] = labels
    adata.obs[cluster_key] = adata.obs[cluster_key].astype('category')

    if not plot:
        return

    cluster_session_orders = leaves_list(Z)
    cluster_colors = [f'C{l - 1}' for l in labels]

    # plot heatmap of distance matrix reordered by clustering result
    g = sns.clustermap(sample_dists[dist_metric], row_linkage=Z,
                        col_linkage=Z, xticklabels=False, yticklabels=False,
                        row_colors=cluster_colors, figsize=(6, 6))
    figure_path = os.path.join(
        sample_cluster_dir,
        f'{dist_metric}_{cluster_method}_distances.pdf')
    g.savefig(figure_path, dpi=300)
    plt.close('all')

    # plot trajectories reordered by clustering result
    g = sns.clustermap(y_sessions[cluster_session_orders, :],
                        row_linkage=Z, col_cluster=False, xticklabels=False,
                        yticklabels=False, row_colors=cluster_colors,
                        figsize=(4, 6), cbar_pos=None)
    g.ax_heatmap.set_xlabel('Ca2+ response')
    g.ax_heatmap.set_ylabel('Cells')
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
                        col_cluster=False, xticklabels=param_names,
                        yticklabels=False, row_colors=cluster_colors,
                        figsize=(4, 6), cbar_pos=None)
    g.ax_heatmap.set_ylabel('Cells')
    plt.tight_layout()
    figure_path = os.path.join(
        sample_cluster_dir,
        f'{dist_metric}_{cluster_method}_posterior_means.pdf')
    g.savefig(figure_path, dpi=300)
    plt.close('all')

# %%
dist_metrics = list(sample_dists.keys())
linkage_mat = {}
cluster_methods = ['single', 'complete', 'average', 'centroid', 'median',
                   'ward']

for metric, method in itertools.product(dist_metrics, cluster_methods):
    print(f'Clustering {metric} using {method} linkage...')
    cluster_by_sample_distances(metric, method, plot=False)

# %%
def cluster_by_sample_stats(stat_type, cluster_method, plot=False):
    if stat_type == 'mean':
        data = sample_means
    else:  # data_type == 'mode'
        data = sample_modes

    if cluster_method == 'kmeans':
        kmeans = KMeans(n_clusters=max_num_clusters, random_state=0)
        labels = kmeans.fit_predict(data)
    else:
        Z = linkage(data, method=cluster_method)
        linkage_mat[(stat_type, cluster_method)] = Z
        labels = fcluster(Z, max_num_clusters, criterion='maxclust')
    cluster_labels[(stat_type, cluster_method)] = labels
    cluster_key = f'posterior_{stat_type}_{cluster_method}'
    adata.obs[cluster_key] = labels
    adata.obs[cluster_key] = adata.obs[cluster_key].astype('category')

    if not plot:
        return

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
        figure_path = os.path.join(sample_cluster_dir,
                                   f'{cluster_key}_trajectories.pdf')
        plt.savefig(figure_path)
        plt.close('all')

        # plot posterior means/modes reordered by clustering result
        plt.figure(figsize=(4, 6), dpi=300)
        _ = sns.heatmap(data[reordered_session_indices, :],
                        xticklabels=param_names, yticklabels=reordered_labels)
        plt.ylabel('Cells')
        plt.tight_layout()
        figure_path = os.path.join(sample_cluster_dir, f'{cluster_key}.pdf')
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
        figure_path = os.path.join(sample_cluster_dir,
                                   f'{cluster_key}_trajectories.pdf')
        plt.savefig(figure_path)
        plt.close('all')

        # plot posterior means reordered by clustering result
        plt.figure(figsize=(4, 6), dpi=300)
        g = sns.clustermap(data, row_linkage=Z, col_cluster=False,
                           xticklabels=param_names, yticklabels=False,
                           row_colors=cluster_colors, figsize=(4, 6),
                           cbar_pos=None)
        g.ax_heatmap.set_ylabel('Cells')
        plt.tight_layout()
        figure_path = os.path.join(sample_cluster_dir, f'{cluster_key}.pdf')
        g.savefig(figure_path, dpi=300)
        plt.close('all')

# %%
# cluster posterior samples using k-means
cluster_methods = ['single', 'complete', 'average', 'centroid', 'median',
                   'ward', 'kmeans']
stat_types = ['mean', 'mode']
for st, method in itertools.product(stat_types, cluster_methods):
    if method != 'kmeans':
        print(f'Clustering posterior {st} using {method} linkage...')
    else:
        print(f'Clustering posterior {st} using k-means...')

    cluster_by_sample_stats(st, method, plot=False)

# %%
# find differential genes in each cluster
cluster_keys = ['kl_yao_ward', 'posterior_mean_kmeans',
                'posterior_mode_kmeans', 'posterior_mean_ward',
                'posterior_mode_ward']
marker_gene_tests = ['t-test', 'wilcoxon', 't-test_overestim_var']
num_top_genes = 10
# p_val_max = 1.1

for cluster_key, test in itertools.product(cluster_keys, marker_gene_tests):
    print(f'Finding marker genes for {cluster_key} using {test}...')
    # get marker genes
    marker_gene_key = f'{cluster_key}_{test}'
    sc.tl.rank_genes_groups(adata, cluster_key, n_genes=num_top_genes,
                            method=test, key_added=marker_gene_key)
    # marker_gene_table = sc.get.rank_genes_groups_df(
    #     adata, None, key=marker_gene_key, pval_cutoff=p_val_max)

    # plot marker genes
    # sc.pl.rank_genes_groups(
    #     adata, n_genes=10, key=marker_gene_key, sharey=False,
    #     save=f'_{metric}_{method}_{num_max_clusters}_{test}.pdf',
    #     show=False)

    g = sc.pl.rank_genes_groups_heatmap(
        adata, n_genes=num_top_genes, groupby=cluster_key,
        key=marker_gene_key, dendrogram=False, use_raw=False, show=False,
        save=f'_{marker_gene_key}_marker_genes.pdf')

    g = sc.pl.rank_genes_groups_violin(
        adata, key=marker_gene_key, use_raw=False, show=False,
        save=f'_{test}.pdf')

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
            cell_samples = analyzer.session_analyzers[cell_idx].get_samples(
                    excluded_params=excluded_params)
            cluster_samples[cluster].loc[i, :] = \
                cell_samples.sample(1, random_seed=bit_generator).values

    # plot sampled parameters
    print(f'Plotting samples...')
    figure_path = os.path.join(sample_cluster_dir,
                               f'{metric}_{method}_sample_hist.pdf')
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

# %%
# plot simulated trajecotries from each cluster
cluster_keys = ['kl_yao_ward', 'posterior_mean_kmeans',
                'posterior_mode_kmeans']
subsample_size = 500
random_seed = 0
bit_generator = np.random.MT19937(random_seed)
rng = np.random.default_rng(bit_generator)

for cluster_key in cluster_keys:
    cluster_names = adata.obs[cluster_key].cat.categories

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
            sample_cluster_dir,
            f'{cluster_key}_simulated_trajectories_cluster_{cluster}.pdf')
        plt.savefig(figure_path)
        plt.close()

# %%
# compute trajectory statistics in each cluster
cluster_keys = ['kl_yao_ward', 'posterior_mean_kmeans',
                'posterior_mode_kmeans', 'posterior_mean_ward',
                'posterior_mode_ward']
cluster_traj_stats = ['PeakMean', 'PeakStd', 'PeakTimeMean', 'PeakTimeStd',
                      'SteadyMean', 'SteadyStd']
cluster_traj_stat_table = {}
num_steady_pts = ts.size // 5

for cluster_key in cluster_keys:
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
        steady_states = np.mean(y_cluster[:, -num_steady_pts:],
                                        axis=1)
        row['SteadyMean'] = np.mean(steady_states)
        row['SteadyStd'] = np.std(steady_states)
        cluster_traj_features['steady_states'].append(steady_states)

        cluster_traj_stat_table[cluster_key].loc[cluster, :] = row

    for feature, values in cluster_traj_features.items():
        plt.figure(figsize=(6, 4), dpi=300)
        plt.hist(values, bins=20, density=True, label=cluster_names)
        plt.legend()

        figure_title = ' '.join(feature.split('_'))
        figure_title = figure_title.capitalize()
        plt.title(figure_title)

        plt.tight_layout()

        figure_path = os.path.join(sample_cluster_dir,
                                   f'{cluster_key}_traj_{feature}.pdf')
        plt.savefig(figure_path)
        plt.close()

# %%
# sample distance vs similarity
soptsc_vars = scipy.io.loadmat(
        '../../result/SoptSC/SoptSC_feature_100/workspace.mat')
similary_mat = soptsc_vars['W'][np.ix_(session_list_int, session_list_int)]
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
