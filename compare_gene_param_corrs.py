# %%
import os
import os.path
import json

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# %%
# initialize
stan_runs = ['const-Be-eta1', '3', 'const-Be-eta1-mixed-all',
             'const-Be-eta1-random-all']
num_top_genes = 30

# load metadata
with open('stan_run_meta.json', 'r') as f:
    stan_run_meta = json.load(f)

# get top genes from gene-param correlation tables for all cell chains
ranked_genes = {}
gene_count = {}
for run in stan_runs:
    run_dir = os.path.join('../../result', stan_run_meta[run]['output_dir'],
                           'multi-sample-analysis')
    if not run.endswith('-all'):
        # for an earlier run, append cell ranges in its folder name
        run_dir += '-0001-0500'
    gene_param_table_path = os.path.join(run_dir, 'genes-vs-params',
                                         'pearson_corrs_sorted.csv')

    gene_param_table = pd.read_csv(gene_param_table_path, index_col=0)
    ranked_genes[run] = gene_param_table['Gene'].unique()[:num_top_genes]

    for gene in ranked_genes[run]:
        if gene in gene_count:
            gene_count[gene] += 1
        else:
            gene_count[gene] = 1

common_genes = [g for g, c in gene_count.items() if c >= 3]
common_genes.sort()

# load marker genes of Leiden clusters
leiden_gene_table_path = '../../result/gene-clustering/' \
                         'leiden_0.50_t-test_marker_genes.csv'
leiden_gene_table = pd.read_csv(leiden_gene_table_path)
leiden_genes = leiden_gene_table['names'].values

# make a table of genes for all cell chains and Leiden marker genes
gene_summary = pd.DataFrame(index=common_genes, columns=['leiden'] + stan_runs,
                            dtype=int)
for gene in gene_summary.index:
    for run in stan_runs:
        gene_summary.loc[gene, run] = int(gene in ranked_genes[run])

    gene_summary.loc[gene, 'leiden'] = int(gene in leiden_genes)

# sort gene table by total occurences
sorted_genes = gene_summary.sum(axis=1).sort_values(kind='stable')
gene_summary = gene_summary.loc[sorted_genes.index, :]
common_genes = gene_summary.index.tolist()

# %%
# plot the table as a binary heatmap
# change font settings
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['font.size'] = 12

# fig, axs = plt.subplots(nrows=1, ncols=2, subplot_kw={'aspect': 'equal'},
#                         figsize=(3, 8), dpi=300)
gs_kw = dict(width_ratios=[1, 4], height_ratios=[1, 25])
fig, axs = plt.subplot_mosaic(
    [['upper', 'upper'], ['lower left', 'lower right']],
    gridspec_kw=gs_kw, figsize=(3, 8))
num_common_genes = len(common_genes)
gene_colormap = plt.get_cmap('Greys').reversed()

# make legend
axs['upper'].set_axis_off()
legend_patches = [mpatches.Patch(color=gene_colormap(0.75)),
                  mpatches.Patch(color=gene_colormap(0.25))]
axs['upper'].legend(
    legend_patches, ['Marker', 'Unassociated'],
    frameon=False)

# plot Leiden marker genes
axs['lower left'].pcolormesh(
    gene_summary.loc[:, 'leiden'].values.reshape((num_common_genes, 1)),
    cmap=gene_colormap, edgecolors='k', vmin=-0.5, vmax=1.5)
axs['lower left'].set_xticks([0.5])
axs['lower left'].set_xticklabels(labels=['Leiden'], rotation='vertical')
axs['lower left'].set_yticks(np.arange(len(common_genes)) + 0.5)
axs['lower left'].set_yticklabels(reversed(common_genes))

# plot genes from cell chains
axs['lower right'].pcolormesh(gene_summary.loc[:, stan_runs].values,
                  cmap=gene_colormap, edgecolors='k', vmin=-0.5, vmax=1.5)
run_pub_names = [stan_run_meta[r]['pub_name'] for r in stan_runs]
run_pub_names = [n.replace('\\$', '$') for n in run_pub_names]
axs['lower right'].set_xticks(np.arange(len(run_pub_names)) + 0.5)
axs['lower right'].set_xticklabels(run_pub_names, rotation='vertical')
axs['lower right'].set_yticks([])
axs['lower right'].set_yticklabels([])

plt.tight_layout()
figure_path = '../../result/top_corr_genes_vs_leiden_genes.pdf'
plt.savefig(figure_path)
plt.close('all')
