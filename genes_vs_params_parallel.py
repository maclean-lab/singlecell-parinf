# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from math import log
import os.path
import itertools
import math
import numpy as np
import scipy.stats
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData

from stan_helpers import StanSessionAnalyzer

# %%
param_names = ['sigma', 'L', 'Katp', 'KoffPLC', 'Vplc', 'Kip3', 'KoffIP3',
               'a', 'dinh', 'Ke', 'Be', 'd1', 'd5', 'epr', 'eta1', 'eta2',
               'eta3', 'c0', 'k3']
num_params = len(param_names)

use_highly_variable_genes = False

# %%
# load gene expression
raw_data = pd.read_csv('../../data/vol_adjusted_genes_transpose.txt', sep='\t')

# perform log normalization
if use_highly_variable_genes:
    # use highly variable genes only
    adata = AnnData(raw_data.T)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)
    gene_symbols = adata.var_names[adata.var['highly_variable']]

    log_data = np.log1p(raw_data.loc[gene_symbols, :].to_numpy())
else:
    # use all genes
    log_data = np.log1p(raw_data.to_numpy())
    gene_symbols = raw_data.index.to_numpy()

# %%
chain_1 = {}
chain_1['cell_list_path'] = os.path.join(
    'cell_lists', 'dfs_feature_100_root_5106_0.000_1.8_reversed_141_1.txt')
chain_1['stan_output_root'] = os.path.join(
    '../../result', 'stan-calcium-model-100-reversed-141-1')

chain_2 = {}
chain_2['cell_list_path'] = os.path.join(
    'cell_lists', 'dfs_feature_100_root_5106_0.000_1.8_reversed_141_2.txt')
chain_2['stan_output_root'] = os.path.join(
    '../../result', 'stan-calcium-model-100-reversed-141-2')

output_dir = os.path.join('../../result/',
                          'stan-calcium-model-100-reversed-141-comparison',
                          'param_scatter')

# %%
def get_sample_means(chain_meta, first_cell, last_cell):
    cell_list = pd.read_csv(chain_meta['cell_list_path'], sep='\t')
    first_cell_order = np.where(cell_list['Cell'] == first_cell)[0][0]
    last_cell_order = np.where(cell_list['Cell'] == last_cell)[0][0]

    sampled_cell_ids = []
    sample_means = {p: [] for p in param_names}
    for cell_order in range(first_cell_order, last_cell_order + 1):
        # load samples
        cell_id = cell_list.loc[cell_order, 'Cell']
        cell_path = os.path.join(chain_meta['stan_output_root'],
                                 f'cell-{cell_id:04d}')
        analyzer = StanSessionAnalyzer(
            cell_path, sample_source='arviz_inf_data', param_names=param_names)

        # compute sample means for cells with mixed chains (R_hat < 4.0)
        cell_sample_means = analyzer.get_sample_means(rhat_upper_bound=4.0)
        if cell_sample_means:
            sampled_cell_ids.append(cell_id)

            for param in param_names:
                sample_means[param].append(cell_sample_means[param])

    chain_meta['sample_means'] = sample_means
    chain_meta['sampled_cell_ids'] = sampled_cell_ids

get_sample_means(chain_1, 141, 464)
get_sample_means(chain_2, 141, 464)
chain_pairs = (chain_1, chain_2)

# %%
from matplotlib.backends.backend_pdf import PdfPages

def scatter_multi_plot(X_data, chains, output_path, num_rows=4,
                       num_cols=2):
    """Make multiple scatter plots in a PDF"""
    num_subplots_per_page = num_rows * num_cols
    num_plots = num_params
    num_pages = math.ceil(num_plots / num_subplots_per_page)

    with PdfPages(output_path) as pdf:
        # generate each page
        for page in range(num_pages):
            # set page size as US letter
            plt.figure(figsize=(8.5, 11))

            # set number of subplots in current page
            if page == num_pages - 1:
                num_subplots = (num_plots - 1) % num_subplots_per_page + 1
            else:
                num_subplots = num_subplots_per_page

            # make each subplot
            for plot_idx in range(num_subplots):
                plt.subplot(num_rows, num_cols, plot_idx + 1)
                param = param_names[page * num_subplots_per_page + plot_idx]
                plt.scatter(X_data[chains[0]['sampled_cell_ids']],
                            chains[0]['sample_means'][param])
                plt.scatter(X_data[chains[1]['sampled_cell_ids']],
                            chains[1]['sample_means'][param])
                plt.title(param)

            plt.tight_layout()
            pdf.savefig()
            plt.close()

for i, gene in enumerate(gene_symbols):
    gene_output_path = os.path.join(output_dir, f'{gene}.pdf')
    scatter_multi_plot(log_data[i, :], chain_pairs, gene_output_path)

# %%
