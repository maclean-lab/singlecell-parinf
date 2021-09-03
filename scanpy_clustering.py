# %%
import os.path
import json

import pandas as pd
import scanpy as sc

# %%
stan_run = 'const-Be-eta1'
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
sc.settings.figdir = output_root

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
sc.tl.leiden(adata, resolution=0.5)
# sc.pl.umap(adata, color='leiden', use_raw=False)

# %%
marker_gene_tests = ['t-test', 'wilcoxon', 't-test_overestim_var']
for test in marker_gene_tests:
    sc.tl.rank_genes_groups(adata, 'leiden', method=test)
    # sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)
    _ = sc.pl.rank_genes_groups_heatmap(
        adata, n_genes=10, groupby='leiden', use_raw=False, show=False,
        save=f'_leiden_marker_genes_{test}.pdf')

# %%
