# %%
import os.path
import json

import pandas as pd
import scanpy as sc

# %%
# stan_run = 'const-Be-eta1'
# stan_run = 'const-Be-eta1-mixed-all'
stan_run = 'const-Be-eta1-random-1'
first_cell_order = 1
last_cell_order = 500

# get cell list
with open('stan_run_meta.json', 'r') as f:
    stan_run_meta = json.load(f)
cell_list_path = os.path.join(
    'cell_lists', stan_run_meta[stan_run]['cell_list'])
full_cell_list = pd.read_csv(cell_list_path, sep='\t')
cell_list = full_cell_list.loc[first_cell_order:last_cell_order, 'Cell']

output_root = os.path.join(
    '../../result', stan_run_meta[stan_run]['output_dir'])
output_dir = os.path.join(output_root, 'gene-clustering')
sc.settings.figdir = output_dir

leiden_resolution = 0.5
num_top_genes = 10

# %%
# load expression data and preprocess
print('Loading gene expression...')
adata = sc.read_csv('../../data/vol_adjusted_genes.csv')
adata = adata[cell_list, :]
adata.raw = adata
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.scale(adata)

# %%
# cluster
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
cluster_key = f'leiden_{leiden_resolution:.2f}'
sc.tl.leiden(adata, resolution=leiden_resolution, key_added=cluster_key)
# sc.pl.umap(adata, color='leiden', use_raw=False)

# %%
marker_gene_tests = ['t-test', 'wilcoxon', 't-test_overestim_var']
for test in marker_gene_tests:
    marker_gene_key = f'{cluster_key}_{test}'
    sc.tl.rank_genes_groups(adata, cluster_key, method=test,
                            key_added=marker_gene_key)
    # sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)
    _ = sc.pl.rank_genes_groups_heatmap(
        adata, n_genes=num_top_genes, key=marker_gene_key, use_raw=False,
        show=False, save=f'_{marker_gene_key}_marker_genes.pdf')

    _ = sc.pl.rank_genes_groups_violin(
        adata, n_genes=10, use_raw=False, key=marker_gene_key, show=False,
        save=f'_{test}.pdf')

# %%
adata.write(
    os.path.join(output_dir, f'leiden_{leiden_resolution:.2f}_adata.h5ad'))

# %%
