# %%
import os
import os.path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
# import umap
import tqdm
import matplotlib.pyplot as plt
from stan_helpers import StanSessionAnalyzer

# %%
# load samples
first_cell_order = 1
last_cell_order = 500
cell_list_path = f'cell_lists/dfs_feature_100_root_5106_0.000_1.8.txt'
cell_list = pd.read_csv(cell_list_path, sep='\t')
cell_list = cell_list.iloc[first_cell_order:last_cell_order + 1, :]

run_dir = '../../result/stan-calcium-model-100-root-5106-3'
all_param_names = ['sigma', 'KonATP', 'L', 'Katp', 'KoffPLC', 'Vplc', 'Kip3',
                   'KoffIP3', 'a', 'dinh', 'Ke', 'Be', 'd1', 'd5', 'epr',
                   'eta1', 'eta2', 'eta3', 'c0', 'k3']
run_param_mask = '0111111111111111111'
param_names = [all_param_names[i + 1] for i, mask in enumerate(run_param_mask)
               if mask == '1']
param_names = ['sigma'] + param_names
analysis_param_mask = '0111111111011101111'
analysis_param_mask = [m == '1' for m in analysis_param_mask]
analysis_param_names = [all_param_names[i + 1]
                        for i, mask in enumerate(analysis_param_mask) if mask]
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

top_param_fractions = [0.9, 0.8, 0.7]
top_param_counts = {}
for fraction in top_param_fractions:
    top_param_counts[fraction] = pd.DataFrame(
        0, columns=analysis_param_names, index=range(num_analysis_params))

# %%
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
    samples = samples.loc[:, analysis_param_mask]
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
