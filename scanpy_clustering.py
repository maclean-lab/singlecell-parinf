# %%
import os.path
import json

import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc

# %%
# stan_runs = ['const-Be-eta1']
# stan_runs = [f'const-Be-eta1-mixed-{i}' for i in range(5)]
stan_runs = [f'const-Be-eta1-random-{i}' for i in range(1, 7)]
# list_ranges = [(1, 500)]
# list_ranges = [(1, 100), (1, 100), (1, 100), (1, 100), (1, 100)]
list_ranges = [(1, 382), (1, 100), (1, 100), (1, 100), (1, 100), (1, 100)]
num_runs = len(stan_runs)

# get cell list
with open('stan_run_meta.json', 'r') as f:
    stan_run_meta = json.load(f)
session_list = []
for run, lr in zip(stan_runs, list_ranges):
    cell_list_path = os.path.join('cell_lists',
                                  stan_run_meta[run]['cell_list'])
    run_cell_list = pd.read_csv(cell_list_path, sep='\t')
    cell_list = run_cell_list.iloc[lr[0]:lr[1] + 1, :]
    run_root = os.path.join('../../result', stan_run_meta[run]['output_dir'])
    session_list.extend([str(c) for c in cell_list['Cell']])

# load all samples
if num_runs == 1:
    output_root = stan_run_meta[stan_runs[0]]['output_dir']
else:
    output_root = stan_run_meta[stan_runs[0]]['output_dir'][:-2] + '-all'
output_root = os.path.join('../../result', output_root)
if not os.path.exists(output_root):
    os.mkdir(output_root)
output_dir = os.path.join(output_root, 'gene-clustering')
sc.settings.figdir = output_dir

leiden_resolution = 1.0

# %%
# load expression data and preprocess
print('Loading gene expression...')
adata = sc.read_csv('../../data/vol_adjusted_genes.csv')
adata = adata[session_list, :]
adata.raw = adata
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.scale(adata)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

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
marker_genes = set()

num_top_genes = 10 if len(adata.obs[cluster_key].cat.categories) < 5 else 5

for test in marker_gene_tests:
    marker_gene_key = f'{cluster_key}_{test}'
    sc.tl.rank_genes_groups(adata, cluster_key, method=test,
                            key_added=marker_gene_key)

    _ = sc.pl.rank_genes_groups_heatmap(
        adata, n_genes=num_top_genes, key=marker_gene_key, use_raw=False,
        show=False, save=f'_{marker_gene_key}_marker_genes.pdf')

    _ = sc.pl.rank_genes_groups_violin(
        adata, n_genes=10, use_raw=False, key=marker_gene_key, show=False,
        save=f'_{test}.pdf')

    for cluster in adata.obs[cluster_key].cat.categories:
        marker_gene_table = sc.get.rank_genes_groups_df(
            adata, cluster, key=marker_gene_key, pval_cutoff=0.05)
        marker_genes.update(marker_gene_table.loc[:5, 'names'])

# %%
# plot genes
with plt.rc_context({"figure.figsize": (4, 4), "figure.dpi": (300)}):
    for gene in marker_genes:
        _ = sc.pl.pca(adata, color=gene, save=f'_{gene}.pdf', show=False)
        _ = sc.pl.umap(adata, color=gene, save=f'_{gene}.pdf', show=False)

# %%
adata.write(
    os.path.join(output_dir, f'leiden_{leiden_resolution:.2f}_adata.h5ad'))

# %%
