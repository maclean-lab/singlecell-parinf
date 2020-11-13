# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os.path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm, trange
import scanpy as sc
from anndata import AnnData

from stan_helpers import StanSessionAnalyzer
from compare_sessions import compare_params

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
print('Loading chain 1...')
chain_1 = {}
chain_1['cell_list_path'] = os.path.join(
    'cell_lists', 'dfs_feature_100_root_5106_0.000_1.8_reversed_141_1.txt')
chain_1['stan_output_root'] = os.path.join(
    '../../result', 'stan-calcium-model-100-reversed-141-1')

print('Loading chain 2...')
chain_2 = {}
chain_2['cell_list_path'] = os.path.join(
    'cell_lists', 'dfs_feature_100_root_5106_0.000_1.8_reversed_141_2.txt')
chain_2['stan_output_root'] = os.path.join(
    '../../result', 'stan-calcium-model-100-reversed-141-2')

print('Loading chain 3...')
chain_3 = {}
chain_3['cell_list_path'] = os.path.join(
    'cell_lists', 'dfs_feature_100_root_5106_0.000_1.8_reversed_141_1.txt')
chain_3['stan_output_root'] = os.path.join(
    '../../result', 'stan-calcium-model-100-reversed-141-3')

output_dir = os.path.join('../../result/',
                          'stan-calcium-model-100-reversed-141-comparison')

# %%
def get_chain_data(chain, first_cell, last_cell):
    cell_list = pd.read_csv(chain['cell_list_path'], sep='\t')
    first_cell_order = np.where(cell_list['Cell'] == first_cell)[0][0]
    last_cell_order = np.where(cell_list['Cell'] == last_cell)[0][0]

    chain['cell_ids'] = []
    chain['analyzers'] = []
    chain['samples'] = []
    chain['sample_means'] = pd.DataFrame(columns=param_names)
    for cell_order in trange(first_cell_order, last_cell_order + 1):
        # load samples
        cell_id = cell_list.loc[cell_order, 'Cell']
        cell_path = os.path.join(chain['stan_output_root'],
                                 f'cell-{cell_id:04d}')
        analyzer = StanSessionAnalyzer(
            cell_path, sample_source='arviz_inf_data', param_names=param_names)

        # compute sample means for cells with mixed chains (R_hat < 4.0)
        cell_samples = analyzer.get_samples(rhat_upper_bound=4.0)
        cell_sample_means = analyzer.get_sample_means(
            rhat_upper_bound=4.0)
        if cell_samples is not None:
            chain['cell_ids'].append(cell_id)
            chain['analyzers'].append(analyzer)
            chain['samples'].append(cell_samples)
            chain['sample_means'] = chain['sample_means'].append(
                cell_sample_means, ignore_index=True)

get_chain_data(chain_1, 141, 464)
get_chain_data(chain_2, 141, 464)
get_chain_data(chain_3, 141, 464)
full_chains = (chain_1, chain_2, chain_3)

# %%
# make violin plots of parameters from every NUTS chain of each cell in both
# runs
session_ids = [1, 2]
chain_list = [[0, 1, 2, 3], [0, 1, 2, 3]]
common_cell_id_set = set()
common_cell_id_set.update(chain_1['cell_ids'])
common_cell_id_set.update(chain_2['cell_ids'])
common_cell_ids = np.intersect1d(chain_1['cell_ids'], chain_2['cell_ids'])

# %%
for cell_id in common_cell_id_set:
    chain_1_cell_order = np.where(chain_1['cell_ids'] == cell_id)[0][0]
    chain_2_cell_order = np.where(chain_2['cell_ids'] == cell_id)[0][0]
    analyzer_1 = chain_1['analyzers'][chain_1_cell_order]
    analyzer_2 = chain_2['analyzers'][chain_2_cell_order]
    output_name = f'param_violin_{cell_id:04d}.pdf'
    compare_params([analyzer_1, analyzer_2], session_ids, chain_list,
                   os.path.join(output_dir, 'param-violins-by-cell'),
                   param_names, output_name=output_name)

# %%
# make scatter plots of parameter means vs expression of every gene
from matplotlib.backends.backend_pdf import PdfPages

def scatter_multi_plot(X_data, chains, output_path, num_rows=4, num_cols=2):
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
                for chain in chains:
                    plt.scatter(X_data[chain['cell_ids']],
                                chain['sample_means'][param], alpha=0.3)
                plt.title(param)

            plt.tight_layout()
            pdf.savefig()
            plt.close()

for i, gene in enumerate(tqdm(gene_symbols)):
    gene_output_path = os.path.join(output_dir, 'gene-vs-param-scatters',
                                    f'{gene}.pdf')
    scatter_multi_plot(log_data[i, :], full_chains, gene_output_path)

# %%
# filter cells mixed in all chains
common_cell_id_set.update(chain_3['cell_ids'])

def filter_cells(chain):
    filtered_chain = {}
    filtered_chain['cell_ids'] = []
    filtered_chain['analyzers'] = []
    filtered_chain['samples'] = []
    is_cell_mixed = [cell_id in common_cell_id_set
                     for cell_id in chain['cell_ids']]

    for i, is_mixed in enumerate(is_cell_mixed):
        if is_mixed:
            filtered_chain['cell_ids'].append(chain['cell_ids'][i])
            filtered_chain['analyzers'].append(chain['analyzers'][i])
            filtered_chain['samples'].append(chain['samples'][i])

    filtered_chain['sample_means'] = chain['sample_means'].loc[is_cell_mixed, :]

    return filtered_chain

filtered_chain_1 = filter_cells(chain_1)
filtered_chain_2 = filter_cells(chain_2)
filtered_chain_3 = filter_cells(chain_3)

common_cell_ids = [i for i in chain_1['cell_ids'] if i in common_cell_id_set]

# %%
# make violin plots for each parameter of both chains in terms of chain
# progression
# only mixed samples are used
def violin_multi_plot(chains, output_path, num_rows=4, num_cols=1,
                      quantile_min=0.1, quantile_max=0.9,
                      canvas_size=(8.5, 11), dpi=100, ylims=None,
                      chain_colors=None):
    """Make multiple violin plots in a PDF"""
    if not isinstance(chains, list) and not isinstance(chains, tuple):
        chains = [chains]
    num_subplots_per_page = num_rows * num_cols
    num_plots = num_params
    num_pages = math.ceil(num_plots / num_subplots_per_page)
    xtick_pos = np.arange(1, len(common_cell_ids) + 1)
    if not chain_colors:
        chain_colors = [f'C{i}' for i in range(len(chains))]
    if not isinstance(chain_colors, list) and \
        not isinstance(chain_colors, list):
        chain_colors = [chain_colors]
    progress_bar = tqdm(total=num_params)

    with PdfPages(output_path) as pdf:
        # generate each page
        for page in range(num_pages):
            # set page size
            plt.figure(figsize=canvas_size, dpi=dpi)

            # set number of subplots in current page
            if page == num_pages - 1:
                num_subplots = (num_plots - 1) % num_subplots_per_page + 1
            else:
                num_subplots = num_subplots_per_page

            # make each subplot
            for plot_idx in range(num_subplots):
                plt.subplot(num_rows, num_cols, plot_idx + 1)
                param = param_names[page * num_subplots_per_page + plot_idx]
                progress_bar.update(1)

                # plot each chain
                for i, chain in enumerate(chains):
                    # get data inside quantiles ranges
                    violin_data = []
                    for s in chain['samples']:
                        s_param = s[param]
                        s_min = s_param.quantile(q=quantile_min)
                        s_max = s_param.quantile(q=quantile_max)
                        s_param = s_param[
                            (s_min <= s_param) & (s_param <= s_max)]
                        violin_data.append(s_param)

                    violin_parts = plt.violinplot(violin_data)
                    if chain_colors is not None:
                        for pc in violin_parts['bodies']:
                            pc.set_facecolor(chain_colors[i])
                            pc.set_edgecolor(chain_colors[i])

                        for p in ['cmins', 'cmaxes', 'cbars']:
                            violin_parts[p].set_facecolor(chain_colors[i])
                            violin_parts[p].set_edgecolor(chain_colors[i])

                plt.xticks(xtick_pos, common_cell_ids, rotation='vertical')
                if ylims is not None:
                    plt.ylim(ylims[param])
                plt.title(param)

            plt.tight_layout()
            pdf.savefig()
            plt.close()

violin_ylims = {'sigma': (0, 0.2), 'L': (0, 0.04), 'Katp': (0, 0.08),
                'KoffPLC': (0, 0.08), 'Vplc': (0, 0.65), 'Kip3': (0, 0.15),
                'KoffIP3': (0, 0.14), 'a': (0, 0.04), 'dinh': (0, 1),
                'Ke': (0, 0.05), 'Be': (0, 160), 'd1': (0, 10), 'd5': (0, 4),
                'epr': (0, 1.2), 'eta1': (570, 600), 'eta2': (0, 1),
                'eta3': (0, 5), 'c0': (0, 100), 'k3': (0, 0.8)}

violin_multi_plot([filtered_chain_1, filtered_chain_2],
                  os.path.join(output_dir, 'param_violin_full.pdf'), num_rows=1,
                  canvas_size=(20, 5), dpi=300)

violin_multi_plot([filtered_chain_1, filtered_chain_2],
                  os.path.join(output_dir, 'param_violin.pdf'), num_rows=1,
                  canvas_size=(20, 5), dpi=300, ylims=violin_ylims)

violin_multi_plot(filtered_chain_1,
                  os.path.join(output_dir, 'param_violin_1.pdf'), num_rows=1,
                  canvas_size=(20, 5), dpi=300, chain_colors='C0',
                  ylims=violin_ylims)

violin_multi_plot(filtered_chain_2,
                  os.path.join(output_dir, 'param_violin_2.pdf'), num_rows=1,
                  canvas_size=(20, 5), dpi=300, chain_colors='C1',
                  ylims=violin_ylims)

violin_multi_plot(filtered_chain_3,
                  os.path.join(output_dir, 'param_violin_3.pdf'), num_rows=1,
                  canvas_size=(20, 5), dpi=300, chain_colors='C2',
                  ylims=violin_ylims)

# %%
# plot fold changes in means for each parameter
def fold_change_multi_plot(chains, output_path, num_rows=4, num_cols=1):
    """Make multiple scatter plots in a PDF"""
    num_subplots_per_page = num_rows * num_cols
    num_plots = num_params
    num_pages = math.ceil(num_plots / num_subplots_per_page)
    xtick_pos = np.arange(1, len(common_cell_ids))
    progress_bar = tqdm(total=num_params, position=0, leave=True)

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
                progress_bar.update(1)

                # get fold changes of sample means in each chain
                for chain in chains:
                    param_mean = chain['sample_means'][param].to_numpy()
                    fold_changes = param_mean[1:] / param_mean[:-1]
                    plt.plot(fold_changes, '-', alpha=0.3)

                plt.xticks(xtick_pos, common_cell_ids[1:], rotation='vertical')
                plt.title(param)

            plt.tight_layout()
            pdf.savefig()
            plt.close()

fold_change_multi_plot((filtered_chain_1, filtered_chain_2, filtered_chain_3),
                       os.path.join(output_dir, 'param_mean_fold_changes.pdf'))

# %%
