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
from stan_helpers import StanSessionAnalyzer

# %%
# load samples
# stan_run = '3'
# stan_run = 'const-Be-eta1'
stan_run = 'const-Be-eta1-mixed-0'
first_cell_order = 1
last_cell_order = 100
analysis_param_mask = '0111111111011101111'

with open('stan_run_meta.json', 'r') as f:
    stan_run_meta = json.load(f)
cell_list_path = os.path.join('cell_lists',
                              stan_run_meta[stan_run]['cell_list'])
cell_list = pd.read_csv(cell_list_path, sep='\t')
cell_list = cell_list.iloc[first_cell_order:last_cell_order + 1, :]

run_dir = os.path.join('../../result', stan_run_meta[stan_run]['output_dir'])
all_param_names = ['sigma', 'KonATP', 'L', 'Katp', 'KoffPLC', 'Vplc', 'Kip3',
                   'KoffIP3', 'a', 'dinh', 'Ke', 'Be', 'd1', 'd5', 'epr',
                   'eta1', 'eta2', 'eta3', 'c0', 'k3']
run_param_mask = stan_run_meta[stan_run]['param_mask']
param_names = [all_param_names[i + 1] for i, mask in enumerate(run_param_mask)
               if mask == '1']
param_names = ['sigma'] + param_names
analysis_param_names = [all_param_names[i + 1]
                        for i, mask in enumerate(analysis_param_mask)
                        if mask == '1']
num_analysis_params = len(analysis_param_names)

pca_dir = os.path.join(run_dir, 'PCA')
if not os.path.exists(pca_dir):
    os.mkdir(pca_dir)

umap_dir = os.path.join(run_dir, 'UMAP')
if not os.path.exists(umap_dir):
    os.mkdir(umap_dir)

mds_dir = os.path.join(run_dir, 'MDS')
if not os.path.exists(mds_dir):
    os.mkdir(mds_dir)

# configure plotting
plot_figures = False
figure_size = (6, 6)
dpi = 300
pc_xtick_locs = np.arange(len(analysis_param_names))
pc_xtick_labels = pc_xtick_locs + 1

# %%
top_param_fractions = [0.9, 0.8, 0.7]
top_param_counts = {}
for fraction in top_param_fractions:
    top_param_counts[fraction] = pd.DataFrame(
        0, columns=analysis_param_names, index=range(num_analysis_params))

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

    # # run UMAP
    # umap_reducer = umap.UMAP()
    # umap_embedding = umap_reducer.fit_transform(samples)
    # plt.figure(figsize=figure_size, dpi=dpi)
    # plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1])
    # plt.savefig(os.path.join(umap_dir, f'{i + 1:04d}_cell_{cell_id:04d}.pdf'))
    # plt.close()

    # # run MDS
    # mds = MDS()
    # mds_transformed_samples = mds.fit_transform(samples)
    # plt.figure(figsize=figure_size, dpi=dpi)
    # plt.scatter(mds_transformed_samples[:, 0], mds_transformed_samples[:, 1])
    # plt.savefig(os.path.join(mds_dir, f'{i + 1:04d}_cell_{cell_id:04d}.pdf'))
    # plt.close()

# %%
def pca_on_cell(base_cell_idx, plot_lemon_prior, norm_func, random_seed=0):
    def normalize_min_max(x, low, high):
        return (x - low) / (high - low)

    def normalize_gaussian(x, mu, sigma):
        return (x - mu) / sigma

    def project_samples(pca_instance, x, norm_func, *args):
        x = norm_func(x, *args)

        if not isinstance(x, np.ndarray):
            x = x.to_numpy()

        if len(x.shape) == 1:
            x = x[np.newaxis, :]

        return pca_instance.transform(x)

    fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
    ax.set_aspect('equal', adjustable='datalim')
    colormap = plt.get_cmap('YlOrRd')

    bit_generator = np.random.MT19937(random_seed)
    # rng = np.random.default_rng(bit_generator)

    # run PCA on as base cell
    if base_cell_idx == 0:
        # base_cell_id = cell_list.loc[1, 'Parent']
        base_cell_id = 5106
        root_dir = os.path.join('../../result',
                                stan_run_meta[stan_run]['root_cell_dir'])
        analyzer = StanSessionAnalyzer(root_dir, param_names=param_names)
    else:
        base_cell_id = cell_list.loc[base_cell_idx, 'Cell']
        cell_dir = os.path.join(run_dir, 'samples', f'cell-{base_cell_id:04d}')
        analyzer = StanSessionAnalyzer(cell_dir, param_names=param_names)
    mixed_chains = analyzer.get_mixed_chains()
    base_samples = pd.concat([analyzer.samples[c] for c in mixed_chains],
                             axis=0, ignore_index=True)
    base_samples = base_samples.sample(n=1000, axis=0,
                                       random_state=bit_generator)
    base_samples = base_samples.loc[:, analysis_param_names]
    # normalize samples to [0, 1]
    base_min, base_max = base_samples.min(axis=0), base_samples.max(axis=0)
    base_mean, base_std = base_samples.mean(axis=0), base_samples.std(axis=0)
    if norm_func == 'min_max':
        base_samples = normalize_min_max(base_samples, base_min, base_max)
    else:
        base_samples = normalize_gaussian(base_samples, base_mean, base_std)
    pca = PCA(n_components=num_analysis_params, svd_solver='full')
    pca.fit(base_samples.to_numpy())

    sampled_cells = cell_list.loc[first_cell_order:last_cell_order,
                                  'Cell'].to_numpy()

    # get similarity matrix
    soptsc_vars = scipy.io.loadmat(
            '../../result/SoptSC/SoptSC_feature_100/workspace.mat')
    similarity_matrix = soptsc_vars['W']
    similarity_matrix /= np.amax(similarity_matrix[base_cell_id, sampled_cells])

    for i, cell_id in enumerate(tqdm.tqdm(sampled_cells)):
        if i == base_cell_idx - 1:
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
            pcs = project_samples(pca, samples, normalize_min_max, base_min,
                                  base_max)
        else:
            pcs = project_samples(pca, samples, normalize_gaussian, base_mean,
                                  base_std)

        sim = similarity_matrix[base_cell_id, cell_id]
        zorder = 2 if sim > 0 else 1
        ax.scatter(pcs[:, 0], pcs[:, 1], s=1, color=colormap(sim),
                   alpha=0.3, edgecolors='none', zorder=zorder)

    pcs = pca.transform(base_samples.to_numpy())
    ax.scatter(pcs[:, 0], pcs[:, 1], s=1, c='C0', alpha=0.3, edgecolors='none',
               zorder=3)

    if plot_lemon_prior:
        if norm_func == 'min_max':
            lemon_prior_on_pca = project_samples(pca, lemon_prior_mu,
                                                 normalize_min_max, base_min,
                                                 base_max)
        else:
            lemon_prior_on_pca = project_samples(pca, lemon_prior_mu,
                                                 normalize_gaussian, base_mean,
                                                 base_std)

        ax.scatter(lemon_prior_on_pca[:, 0], lemon_prior_on_pca[:, 1], c='k',
                   marker='x')

        figure_path = os.path.join(
            pca_proj_dir,
            f'samples_on_{base_cell_idx:04d}_pca_{norm_func}_vs_lemon.png')
    else:
        figure_path = os.path.join(
            pca_proj_dir, f'samples_on_{base_cell_idx:04d}_pca_{norm_func}.png')

    fig.savefig(figure_path)
    plt.close()

pca_proj_dir = os.path.join(run_dir, 'PCA-projection')
if not os.path.exists(pca_proj_dir):
    os.mkdir(pca_proj_dir)

lemon_prior = pd.read_csv('stan_models/equiv_2/calcium_model_alt_prior.txt',
                          sep='\t', index_col=0)
lemon_prior_mu = lemon_prior.loc[analysis_param_names, 'mu']

# plot all samples on PCs of root cell
# base_cell_indices = [0, 1, 101, 201, 301, 401]
base_cell_indices = [0, 1, 21, 41, 61, 81]
show_lemon_prior = [True, False]
norm_funcs = ['min_max', 'gaussian']
for idx, show, func in itertools.product(
        base_cell_indices, show_lemon_prior, norm_funcs):
    pca_on_cell(idx, show, func, random_seed=0)

# %%