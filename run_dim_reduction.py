# %%
import os
import os.path
import itertools
import json

import numpy as np
import scipy.io
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import umap
import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl

from stan_helpers import StanSessionAnalyzer, StanMultiSessionAnalyzer
import calcium_models

# %%
# load samples
# stan_run = '3'
# stan_run = 'const-Be-eta1'
stan_run = 'const-Be-eta1-random-1'
first_cell_order = 1
last_cell_order = 500
analysis_param_mask = '0111111111011101111'

with open('stan_run_meta.json', 'r') as f:
    stan_run_meta = json.load(f)
cell_list_path = os.path.join('cell_lists',
                              stan_run_meta[stan_run]['cell_list'])
cell_list = pd.read_csv(cell_list_path, sep='\t')
cell_list = cell_list.iloc[first_cell_order:last_cell_order + 1, :]
sampled_cells = cell_list.loc[first_cell_order:last_cell_order,
                              'Cell'].to_numpy()
root_cell_dir = os.path.join('../../result',
                             stan_run_meta[stan_run]['root_cell_dir'])
if 'Parent' in cell_list.columns:
    root_cell_id = cell_list.loc[1, 'Parent']
else:
    root_cell_id = int(root_cell_dir[-4:])

run_dir = os.path.join('../../result', stan_run_meta[stan_run]['output_dir'])
run_param_mask = stan_run_meta[stan_run]['param_mask']
param_names = [calcium_models.param_names[i + 1]
               for i, mask in enumerate(run_param_mask) if mask == '1']
param_names = ['sigma'] + param_names
analysis_param_names = [calcium_models.param_names[i + 1]
                        for i, mask in enumerate(analysis_param_mask)
                        if mask == '1']
num_analysis_params = len(analysis_param_names)

soptsc_vars = scipy.io.loadmat(
    '../../result/SoptSC/SoptSC_feature_100/workspace.mat')

# configure plotting
plot_figures = False
figure_size = (6, 6)
dpi = 300
pc_xtick_locs = np.arange(len(analysis_param_names))
pc_xtick_labels = pc_xtick_locs + 1

# change font settings
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['font.size'] = 12

def normalize_min_max(x, low, high):
    return (x - low) / (high - low)

def normalize_gaussian(x, mu, sigma):
    return (x - mu) / sigma

def project_samples(transformer_instance, x, norm_func, *args):
    x = norm_func(x, *args)

    if not isinstance(x, np.ndarray):
        x = x.to_numpy()

    if len(x.shape) == 1:
        x = x[np.newaxis, :]

    return transformer_instance.transform(x)

# %%
# visualize parameters on reduced dimensions for individual cells
top_param_fractions = [0.9, 0.8, 0.7]
top_param_counts = {}
for fraction in top_param_fractions:
    top_param_counts[fraction] = pd.DataFrame(
        0, columns=analysis_param_names, index=range(num_analysis_params))

pca_dir = os.path.join(run_dir, 'PCA')
if not os.path.exists(pca_dir):
    os.mkdir(pca_dir)

umap_dir = os.path.join(run_dir, 'UMAP')
if not os.path.exists(umap_dir):
    os.mkdir(umap_dir)

mds_dir = os.path.join(run_dir, 'MDS')
if not os.path.exists(mds_dir):
    os.mkdir(mds_dir)

for i, cell_id in enumerate(tqdm.tqdm(cell_list['Cell'])):
    cell_dir = os.path.join(run_dir, 'samples', f'cell-{cell_id:04d}')
    analyzer = StanSessionAnalyzer(cell_dir, param_names=param_names)

    # choose 1000 samples from mixed chains for parameters of interest
    mixed_chains = analyzer.get_mixed_chains()
    if not mixed_chains:
        continue

    samples = pd.concat([analyzer.samples[c] for c in mixed_chains], axis=0,
                        ignore_index=True)
    samples = samples.sample(n=1000, axis=0)
    samples = samples.loc[:, analysis_param_names]
    # normalize samples to [0, 1]
    sample_min, sample_max = samples.min(axis=0), samples.max(axis=0)
    samples = (samples - sample_min) / (sample_max - sample_min)

    # run PCA
    pca = PCA(n_components=num_analysis_params)
    pcs = pca.fit_transform(samples)
    pca_loadings = pca.components_.T

    if plot_figures:
        # plot first 2 principal components
        plt.figure(figsize=figure_size, dpi=dpi)
        plt.scatter(pcs[:, 0], pcs[:, 1])
        plt.savefig(
            os.path.join(pca_dir, f'{i + 1:04d}_cell_{cell_id:04d}_2d.pdf'))
        plt.close()

        # plot ratio of variance explained
        plt.figure(figsize=figure_size)
        plt.plot(pca.explained_variance_ratio_, '.')
        plt.xticks(ticks=pc_xtick_locs, labels=pc_xtick_labels)
        plt.savefig(os.path.join(
            pca_dir, f'{i + 1:04d}_cell_{cell_id:04d}_var_explained.pdf'))
        plt.close()

        # plot top parameters
        plt.figure(figsize=figure_size)
        ax = plt.gca()
        heatmap = ax.imshow(np.abs(pca_loadings))
        # set components indices on x-axis
        ax.set_xticks(pc_xtick_locs)
        ax.set_xticklabels(labels=pc_xtick_labels)
        # set parameter names on y-axis
        ax.set_yticks(np.arange(len(analysis_param_names)))
        ax.set_yticklabels(labels=analysis_param_names)
        # add color bar
        plt.colorbar(heatmap)
        plt.savefig(os.path.join(
            pca_dir, f'{i + 1:04d}_cell_{cell_id:04d}_loadings.pdf'))
        plt.close()

    # increment count for parameters with high loadings values in each component
    for comp in range(num_analysis_params):
        max_loadings = np.amax(pca_loadings[:, comp])

        for fraction in top_param_fractions:
            param_threshold = max_loadings * fraction
            for param_idx in range(num_analysis_params):
                if pca_loadings[param_idx, comp] >= param_threshold:
                    top_param_counts[fraction].iloc[comp, param_idx] += 1

    # run UMAP
    umap_reducer = umap.UMAP()
    umap_embedding = umap_reducer.fit_transform(samples)
    plt.figure(figsize=figure_size, dpi=dpi)
    plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1])
    plt.savefig(os.path.join(umap_dir, f'{i + 1:04d}_cell_{cell_id:04d}.pdf'))
    plt.close()

    # run MDS
    mds = MDS()
    mds_transformed_samples = mds.fit_transform(samples)
    plt.figure(figsize=figure_size, dpi=dpi)
    plt.scatter(mds_transformed_samples[:, 0], mds_transformed_samples[:, 1])
    plt.savefig(os.path.join(mds_dir, f'{i + 1:04d}_cell_{cell_id:04d}.pdf'))
    plt.close()

# %%
# project cells onto PC spaces of a focal cell
def project_on_cell(method, base_cell_idx, plot_lemon_prior, norm_func,
                    output_dir, random_seed=0):
    def get_mean_dist(x):
        return np.mean(np.linalg.norm(x, axis=1))

    # initialize
    bit_generator = np.random.MT19937(random_seed)
    mpl.rcParams['font.size'] = 14

    if base_cell_idx == 0:
        base_cell_id = root_cell_id
        analyzer = StanSessionAnalyzer(root_cell_dir, param_names=param_names)
    else:
        base_cell_id = cell_list.loc[base_cell_idx, 'Cell']
        cell_dir = os.path.join(run_dir, 'samples', f'cell-{base_cell_id:04d}')
        analyzer = StanSessionAnalyzer(cell_dir, param_names=param_names)

    proj_stats = pd.DataFrame(columns=['Cell_ID', 'Similarity', 'Distance'])

    # get similarity matrix
    similarity_matrix = soptsc_vars['W']
    similarity_matrix /= np.amax(similarity_matrix[base_cell_id, sampled_cells])

    # set up plotting
    # fig, ax = plt.subplots(figsize=(5, 4), dpi=dpi)
    fig, ax = plt.subplots(figsize=(4, 3.2), dpi=dpi)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_box_aspect(1)
    colormap = plt.get_cmap('YlOrRd')

    # run projection on base cell
    mixed_chains = analyzer.get_mixed_chains()
    base_samples = pd.concat([analyzer.samples[c] for c in mixed_chains],
                             axis=0, ignore_index=True)
    base_samples = base_samples.sample(n=1000, axis=0,
                                       random_state=bit_generator)
    base_samples = base_samples.loc[:, analysis_param_names]
    # normalize samples
    base_min, base_max = base_samples.min(axis=0), base_samples.max(axis=0)
    base_mean, base_std = base_samples.mean(axis=0), base_samples.std(axis=0)
    if norm_func == 'min_max':
        base_samples = normalize_min_max(base_samples, base_min, base_max)
    else:
        base_samples = normalize_gaussian(base_samples, base_mean, base_std)

    if method == 'umap':
        transformer = umap.UMAP()
    elif method == 'mds':
        transformer = MDS()
    else:
        transformer = PCA(n_components=num_analysis_params, svd_solver='full')
    projections = transformer.fit_transform(base_samples.to_numpy())
    ax.scatter(projections[:, 0], projections[:, 1], s=1, c='C0', alpha=0.3,
               edgecolors='none', zorder=3)

    # add stats for base cell
    stats_row = {'Cell_ID': base_cell_id,
                 'Similarity': similarity_matrix[base_cell_id, base_cell_id],
                 'Distance': get_mean_dist(projections[:, :2])}
    proj_stats = proj_stats.append(stats_row, ignore_index=True)

    for cell_id in tqdm.tqdm(sampled_cells):
        if cell_id == base_cell_id:
            continue

        cell_dir = os.path.join(run_dir, 'samples', f'cell-{cell_id:04d}')
        analyzer = StanSessionAnalyzer(cell_dir, param_names=param_names)

        # choose 1000 samples from mixed chains for parameters of interest
        mixed_chains = analyzer.get_mixed_chains()
        if not mixed_chains:
            continue

        samples = pd.concat([analyzer.samples[c] for c in mixed_chains], axis=0,
                            ignore_index=True)
        samples = samples.sample(n=1000, axis=0, random_state=bit_generator)
        samples = samples.loc[:, analysis_param_names]

        if norm_func == 'min_max':
            projections = project_samples(
                transformer, samples, normalize_min_max, base_min, base_max)
        else:
            projections = project_samples(
                transformer, samples, normalize_gaussian, base_mean, base_std)

        sim = similarity_matrix[base_cell_id, cell_id]
        stats_row = {'Cell_ID': cell_id, 'Similarity': sim,
                     'Distance': get_mean_dist(projections[:, :2])}
        proj_stats = proj_stats.append(stats_row, ignore_index=True)

        zorder = 2 if sim > 0 else 1

        if sim > 0:
            ax.scatter(projections[:, 0], projections[:, 1], s=1,
                       color=colormap(sim), alpha=0.3, edgecolors='none',
                       zorder=zorder)
        else:
            ax.scatter(projections[:, 0], projections[:, 1], s=1,
                       color='0.5', alpha=0.3, edgecolors='none',
                       zorder=zorder)

    if plot_lemon_prior:
        if norm_func == 'min_max':
            lemon_prior_projection = project_samples(
                transformer, lemon_prior_mu, normalize_min_max, base_min,
                base_max)
        else:
            lemon_prior_projection = project_samples(
                transformer, lemon_prior_mu, normalize_gaussian, base_mean,
                base_std)

        ax.scatter(lemon_prior_projection[:, 0], lemon_prior_projection[:, 1],
                   c='k', marker='x')

        figure_path = os.path.join(
            output_dir,
            f'samples_on_{base_cell_idx:04d}_{norm_func}_vs_lemon.png')
    else:
        figure_path = os.path.join(
            output_dir,
            f'samples_on_{base_cell_idx:04d}_{norm_func}.png')

    # add axis labels
    if method == 'pca':
        percent_var = transformer.explained_variance_ratio_[0] * 100
        x_label = f'PC1 ({percent_var:.1f}% explained)'
    else:
        x_label = 'PC1'
    ax.set_xlabel(x_label)

    if method == 'pca':
        percent_var = transformer.explained_variance_ratio_[1] * 100
        y_label = f'PC2 ({percent_var:.1f}% explained)'
    else:
        y_label = 'PC2'
    ax.set_ylabel(y_label)

    # show 0 only on each axis
    ax.set_xticks([0])
    ax.set_yticks([0])

    ax.set_title(f'Cell {base_cell_id:04d}')

    # add legend
    # legend_handles = [Line2D([0], [0], linestyle='', color='C0', marker='.',
    #                          label='Focal', markersize=10),
    #                   Line2D([0], [0], linestyle='', color='0.5', marker='.',
    #                          label='Dissimilar', markersize=10)]
    # ax.legend(handles=legend_handles, loc='upper right', fontsize=12)

    # add colorbar
    colormap_norm = mpl.colors.Normalize(vmin=0, vmax=1)
    fig.colorbar(mpl.cm.ScalarMappable(norm=colormap_norm, cmap=colormap),
                 ax=ax, label='Cell-cell similarity')

    plt.tight_layout()
    fig.savefig(figure_path, transparent=True)
    plt.close()

    return proj_stats

lemon_prior = pd.read_csv('stan_models/equiv_2/calcium_model_alt_prior.txt',
                          sep='\t', index_col=0)
lemon_prior_mu = lemon_prior.loc[analysis_param_names, 'mu']

# plot all samples on PCs of root cell
base_cell_indices = [0, 1, 101]#, 201, 301, 401]
# base_cell_indices = [0, 1, 21, 41, 61, 81]
# base_cell_indices = [0]
projection_methods = ['pca']
show_lemon_prior = [False]
norm_funcs = ['min_max', 'gaussian']
proj_stats_all = {}

proj_dir = {}
proj_dir['pca'] = os.path.join(run_dir, 'PCA-projection')
if 'pca' in projection_methods and not os.path.exists(proj_dir['pca']):
    os.mkdir(proj_dir['pca'])
proj_dir['umap'] = os.path.join(run_dir, 'UMAP-projection')
if 'umap' in projection_methods and not os.path.exists(proj_dir['umap']):
    os.mkdir(proj_dir['umap'])
proj_dir['mds'] = os.path.join(run_dir, 'MDS-projection')
if 'mds' in projection_methods and not os.path.exists(proj_dir['mds']):
    os.mkdir(proj_dir['mds'])

for m, idx, show, func in itertools.product(
        projection_methods, base_cell_indices, show_lemon_prior, norm_funcs):
    stats = project_on_cell(m, idx, show, func, proj_dir[m], random_seed=0)
    if not show:
        proj_stats_all[(m, idx, func)] = stats

# %%
mpl.rcParams['font.size'] = 14

for m, idx, show, func in itertools.product(
        projection_methods, base_cell_indices, show_lemon_prior, norm_funcs):
    # plot standalone legend
    plt.figure(figsize=(3.2, 0.5), dpi=300)
    plt.axis('off')
    legend_handles = [
        Line2D([0], [0], linestyle='', color='C0', marker='.',label='Focal',
               markersize=10),
        Line2D([0], [0], linestyle='', color='0.5', marker='.',
               label='Dissimilar', markersize=10)
    ]
    plt.legend(legend_handles, ['Focal', 'Dissimilar'], loc='center',
               frameon=False, bbox_to_anchor=(0.5, 0.5), ncol=2,
               markerscale=1.6)
    plt.tight_layout()
    figure_path = os.path.join(proj_dir[m], 'sample_proj_legend.pdf')
    plt.savefig(figure_path)
    plt.close()

# %%
# plot stats from projection
mpl.rcParams['font.size'] = 14

for (m, idx, func), stats in proj_stats_all.items():
    dists_similar = stats['Distance'][stats['Similarity'] > 0]
    num_similar = dists_similar.size
    dists_dissimilar = stats['Distance'][stats['Similarity'] == 0]
    # dists_dissimilar = dists_dissimilar.sample(n=num_similar, random_state=0)
    x_max = max(dists_similar.max(), dists_dissimilar.max())

    plt.figure(figsize=(3.2, 3.2), dpi=300)
    plt.hist(dists_similar, bins=10, range=(0, x_max), density=True, alpha=0.5,
             label='Similar')
    plt.hist(dists_dissimilar, bins=10, range=(0, x_max), density=True,
             alpha=0.5, label='Dissimilar')
    plt.xlabel('Distance from origin')
    plt.yticks([])
    plt.ylabel('Frequency')
    plt.legend()
    if idx == 0:
        cell_id = root_cell_id
    else:
        cell_id = cell_list.loc[idx, 'Cell']
    plt.title(f'Cell {cell_id:04d}')
    plt.tight_layout()
    figure_path = os.path.join(proj_dir[m],
                               f'samples_on_{idx:04d}_{func}_dists.pdf')
    plt.savefig(figure_path, transparent=True)
    plt.close()

# %%
# plot locations of similar cells
plot_cells = sampled_cells[0:250]
plot_cells = np.insert(plot_cells, 0, root_cell_id)
focal_cell_order = 101

similarity_matrix = soptsc_vars['W'][np.ix_(plot_cells, plot_cells)]
similarity_matrix = np.ceil(similarity_matrix)
num_plot_cells = plot_cells.size
for i, j in itertools.combinations(range(num_plot_cells), 2):
    similarity_matrix[i, j] = 0.5

mpl.rcParams['font.size'] = 14
plt.figure(figsize=(4, 4), dpi=300)
plt.imshow(similarity_matrix, cmap=plt.get_cmap('binary'))

# add red line for focal cell
horizontal_x = np.arange(focal_cell_order + 1)
horizontal_y = np.ones(focal_cell_order + 1) * focal_cell_order
plt.plot(horizontal_x, horizontal_y, color='r', linewidth=0.5)

vertical_x = np.ones(num_plot_cells - focal_cell_order) * focal_cell_order
vertical_y = np.arange(focal_cell_order, num_plot_cells)
plt.plot(vertical_x, vertical_y, color='r', linewidth=0.5)

plt.xlabel('Position in cell chain')
plt.ylabel('Position in cell chain')

figure_title = stan_run_meta[stan_run]['pub_name'].replace('\\$', '$')
plt.title(figure_title)
plt.tight_layout()
figure_path = os.path.join(run_dir, 'PCA-projection', 'similar_cells.pdf')
plt.savefig(figure_path)
plt.close('all')

# %%
# compute distance between adjacent cells
random_seed = 0
bit_generator = np.random.MT19937(random_seed)
pca_stats = pd.DataFrame(columns=['CellID', 'Distance', 'Similarity'])
norm_func = 'gaussian'
proj_dir = os.path.join(run_dir, 'PCA-projection-neighbor')
if not os.path.exists(proj_dir):
    os.mkdir(proj_dir)

# run PCA for root cell
prev_cell_idx = 0
# load sample
analyzer = StanSessionAnalyzer(root_cell_dir, param_names=param_names)
prev_cell_id = root_cell_id
mixed_chains = analyzer.get_mixed_chains()
prev_sample = pd.concat([analyzer.samples[c] for c in mixed_chains],
                        axis=0, ignore_index=True)
prev_sample = prev_sample.sample(n=1000, axis=0, random_state=bit_generator)
prev_sample = prev_sample.loc[:, analysis_param_names]
# normalize sample
if norm_func == 'min_max':
    prev_min, prev_max = prev_sample.min(axis=0), prev_sample.max(axis=0)
    prev_sample_normalized = normalize_min_max(prev_sample, prev_min, prev_max)
else:
    prev_mean, prev_std = prev_sample.mean(axis=0), prev_sample.std(axis=0)
    prev_sample_normalized = normalize_gaussian(prev_sample, prev_mean,
                                                prev_std)
# run PCA
pca = PCA(n_components=num_analysis_params)
prev_sample_projected = pca.fit_transform(prev_sample_normalized)
dist_row = {'CellID': prev_cell_id, 'Distance': np.nan, 'Similarity': np.nan}
pca_stats = pca_stats.append(dist_row, ignore_index=True)

# plot ratio of variance explained for the root cell
plt.figure(figsize=figure_size, dpi=dpi)
plt.plot(pca.explained_variance_ratio_, marker='.', linestyle='none')
plt.tight_layout()
figure_path = os.path.join(
    proj_dir, f'cell_{prev_cell_id:04d}_var_explained_ratio.pdf')
plt.savefig(figure_path)
plt.close()

for cell_id in tqdm.tqdm(sampled_cells):
    # get a subsample of size 1000 from mixed chains of current cell
    cell_dir = os.path.join(run_dir, 'samples', f'cell-{cell_id:04d}')
    analyzer = StanSessionAnalyzer(cell_dir, param_names=param_names)

    mixed_chains = analyzer.get_mixed_chains()
    if not mixed_chains:
        continue

    curr_sample = pd.concat([analyzer.samples[c] for c in mixed_chains],
                            axis=0, ignore_index=True)
    curr_sample = curr_sample.sample(n=1000, axis=0,
                                     random_state=bit_generator)
    curr_sample = curr_sample.loc[:, analysis_param_names]
    if norm_func == 'min_max':
        curr_sample_normalized = normalize_min_max(curr_sample, prev_min,
                                                   prev_max)
    else:
        curr_sample_normalized = normalize_gaussian(curr_sample, prev_mean,
                                                    prev_std)

    # project current cell on previous cell's principal components
    curr_sample_projected = pca.transform(curr_sample_normalized)

    # compute distance between projected samples
    dist_row = {'CellID': cell_id,
                'Distance': np.mean(np.linalg.norm(curr_sample, axis=1)),
                'Similarity': soptsc_vars['W'][prev_cell_id, cell_id]}
    pca_stats = pca_stats.append(dist_row, ignore_index=True)

    # plot projected samples
    plt.figure(figsize=figure_size, dpi=dpi)
    plt.scatter(prev_sample_projected[:, 0], prev_sample_projected[:, 1],
                label='Previous cell')
    plt.scatter(curr_sample_projected[:, 0], curr_sample_projected[:, 1],
                label='Current cell')
    plt.xlabel('PC1')
    plt.xlabel('PC2')
    plt.legend()
    plt.tight_layout()
    figure_path = os.path.join(
        proj_dir,
        f'cell_{cell_id:04d}_on_cell_{prev_cell_id:04d}_{norm_func}.pdf')
    plt.savefig(figure_path)
    plt.close()

    # run PCA on current cell
    if norm_func == 'min_max':
        prev_min, prev_max = curr_sample.min(axis=0), curr_sample.max(axis=0)
        prev_sample_normalized = normalize_min_max(curr_sample, prev_min,
                                                   prev_max)
    else:
        prev_mean, prev_std = curr_sample.mean(axis=0), curr_sample.std(axis=0)
        prev_sample_normalized = normalize_gaussian(curr_sample, prev_mean,
                                                    prev_std)
    pca = PCA(n_components=num_analysis_params)
    prev_sample_projected = pca.fit_transform(prev_sample_normalized)
    prev_cell_id = cell_id

    # plot ratio of variance explained
    plt.figure(figsize=figure_size, dpi=dpi)
    plt.plot(pca.explained_variance_ratio_, marker='.', linestyle='none')
    plt.xlabel('PC')
    plt.ylabel('Ratio of variance explained')
    plt.tight_layout()
    figure_path = os.path.join(
        proj_dir, f'cell_{cell_id:04d}_var_explained_ratio_{norm_func}.pdf')
    plt.savefig(figure_path)
    plt.close()

output_path = os.path.join(proj_dir, f'pca_stats_{norm_func}.csv')
pca_stats.to_csv(output_path)

# summarize PCA results
# plot histogram of PCA distances
plt.figure(figsize=figure_size, dpi=dpi)
plt.hist(pca_stats['Distance'])
plt.xlabel('Distance')
plt.ylabel('Number of cell pairs')
plt.tight_layout()
figure_path = os.path.join(proj_dir, f'distance_hist_{norm_func}.pdf')
plt.savefig(figure_path)
plt.close()

# plot distance vs similarity
plt.figure(figsize=figure_size, dpi=dpi)
plt.scatter(pca_stats['Distance'], pca_stats['Similarity'])
plt.xlabel('Distance')
plt.ylabel('Similarity')
plt.tight_layout()
figure_path = os.path.join(proj_dir, f'distance_vs_similarity_{norm_func}.pdf')
plt.savefig(figure_path)
plt.close()

# %%
session_list = [str(c) for c in cell_list['Cell']]
session_dirs = [os.path.join(run_dir, 'samples', f'cell-{c:04d}')
                for c in cell_list['Cell']]
analyzer = StanMultiSessionAnalyzer(
    session_list, os.path.join(run_dir, 'multi-sample-analysis'),
    session_dirs, param_names=param_names)
root_analyzer = StanSessionAnalyzer(root_cell_dir, param_names=param_names)

# %%
def project_on_cell_mean(base_cell_idx, norm_func, output_dir, random_seed=0):
    def get_mean_dist(x):
        return np.mean(np.linalg.norm(x, axis=1))

    # initialize
    bit_generator = np.random.MT19937(random_seed)

    if base_cell_idx == 0:
        base_cell_id = root_cell_id
    else:
        base_cell_id = cell_list.loc[base_cell_idx, 'Cell']

    # get similarity matrix
    similarity_matrix = soptsc_vars['W']
    similarity_matrix /= np.amax(similarity_matrix[base_cell_id, sampled_cells])

    # set up plotting
    fig, ax = plt.subplots(figsize=(5, 4), dpi=dpi)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_box_aspect(1)
    colormap = plt.get_cmap('YlOrRd')

    # run PCA on sample means
    analyzer.get_sample_means()
    root_sample_means = root_analyzer.get_sample_means()

    sample_means = pd.concat(
        [pd.DataFrame(root_sample_means).T, analyzer.sample_means],
        ignore_index=True)
    sample_means = sample_means.loc[:, analysis_param_names]
    sample_mean_min = np.amin(sample_means, axis=0)
    sample_mean_max = np.amax(sample_means, axis=0)
    sample_mean_mean = np.mean(sample_means, axis=0)
    sample_mean_std = np.std(sample_means, axis=0, ddof=1)

    if norm_func == 'min_max':
        sample_means = normalize_min_max(sample_means, sample_mean_min,
                                         sample_mean_max)
    else:
        sample_means = normalize_gaussian(sample_means, sample_mean_mean,
                                          sample_mean_std)

    transformer = PCA(n_components=num_analysis_params, svd_solver='full')
    projections = transformer.fit_transform(sample_means)

    # plot each cell on the first two PCs
    for i, cell_id in enumerate(np.insert(sampled_cells, 0, root_cell_id)):
        sim = similarity_matrix[base_cell_id, cell_id]

        if cell_id == base_cell_id:
            cell_color = 'C0'
        elif sim > 0:
            cell_color = colormap(sim)
        else:
            cell_color = '0.5'
        zorder = 2 if sim > 0 else 1

        plt.scatter(projections[i, 0], projections[i, 1], color=cell_color,
                    alpha=0.5, zorder=zorder)

    figure_path = os.path.join(
        output_dir, f'samples_means_on_{base_cell_idx:04d}_{norm_func}.png')
    fig.savefig(figure_path)
    plt.close('all')

# %%
for focal_cell_id in range(0, 501, 50):
    project_on_cell_mean(
        focal_cell_id, 'gaussian', os.path.join(run_dir, 'PCA-projection'))
