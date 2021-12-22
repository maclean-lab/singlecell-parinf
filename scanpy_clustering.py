# %%
import os.path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

from stan_helpers import load_trajectories

# %%
# stan_runs = ['3']
stan_runs = ['const-Be-eta1']
# stan_runs = ['const-Be-eta1-mixed-1']
# stan_runs = [f'const-Be-eta1-mixed-{i}' for i in range(5)]
# stan_runs = ['const-Be-eta1-random-2']
# stan_runs = [f'const-Be-eta1-random-{i}' for i in range(1, 7)]
list_ranges = [(1, 500)]
# list_ranges = [(1, 100)]
# list_ranges = [(1, 100), (1, 100), (1, 100), (1, 100), (1, 100)]
# list_ranges = [(1, 372)]
# list_ranges = [(1, 571), (1, 372), (1, 359), (1, 341), (1, 335), (1, 370)]

leiden_resolution = 0.5
cluster_key = f'leiden_{leiden_resolution:.2f}'

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

session_list_int = [int(s) for s in session_list]

# get calcium response
t0 = 200
t_downsample = 300
y_all, y0_all, ts = load_trajectories(t0, filter_type='moving_average',
    moving_average_window=20, downsample_offset=t_downsample)
y_sessions = y_all[session_list_int, :]

# load all samples
num_runs = len(stan_runs)

if num_runs == 1:
    output_root = stan_run_meta[stan_runs[0]]['output_dir']
else:
    output_root = stan_run_meta[stan_runs[0]]['output_dir'][:-2] + '-all'
output_root = os.path.join('../../result', output_root)
if not os.path.exists(output_root):
    os.mkdir(output_root)
output_dir = os.path.join(output_root, 'gene-clustering')
sc.settings.figdir = output_dir

# %%
# load expression data and preprocess
print('Loading gene expression...')
adata = sc.read_csv('../../data/vol_adjusted_genes.csv')
adata = adata[session_list, :]
adata.raw = adata
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.scale(adata)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
# cluster
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=leiden_resolution, key_added=cluster_key)

with plt.rc_context({"figure.figsize": (4, 4), "figure.dpi": (300)}):
    sc.pl.pca(adata, color=cluster_key, use_raw=False, show=False,
            save=f'_{cluster_key}.pdf')
    sc.pl.umap(adata, color=cluster_key, use_raw=False, show=False,
            save=f'_{cluster_key}.pdf')

# %%
marker_gene_tests = ['t-test', 'wilcoxon', 't-test_overestim_var']
marker_genes = set()
p_val_max = 1.1

num_top_genes = 10 if len(adata.obs[cluster_key].cat.categories) < 5 else 5

for test in marker_gene_tests:
    marker_gene_key = f'{cluster_key}_{test}'
    sc.tl.rank_genes_groups(adata, cluster_key, n_genes=num_top_genes,
                            method=test, key_added=marker_gene_key)

    marker_gene_table = sc.get.rank_genes_groups_df(
        adata, None, key=marker_gene_key, pval_cutoff=p_val_max)
    marker_gene_table_path = os.path.join(
        output_dir, f'{marker_gene_key}_marker_genes.csv')
    marker_gene_table.to_csv(marker_gene_table_path)

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
# perform GO analysis for marker genes
from goatools.base import download_go_basic_obo
from goatools.obo_parser import GODag
from goatools.base import download_ncbi_associations
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.test_data.genes_NCBI_9606_ProteinCoding import GENEID2NT \
    as GeneID2nt
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS

# initialize GO analysis
marker_gene_key = f'{cluster_key}_t-test'
filter_col = 'pvals'
# filter_col = 'pvals_adj'
marker_gene_max_pval = 0.05
go_max_pval = 0.05

obo_path = download_go_basic_obo()
obo_dag = GODag(obo_path)
gene2go_path = download_ncbi_associations()
gene_annoation = Gene2GoReader(gene2go_path, taxids=[9606])
ns2assoc = gene_annoation.get_ns2assc()
go_study = GOEnrichmentStudyNS(GeneID2nt.keys(), ns2assoc, obo_dag,
                               propagate_counts = False, alpha = go_max_pval,
                               methods = ['fdr_bh'])

# create reverse mapping from gene symbols to gene IDs
symbol2id = {}
for gene_id, nt in GeneID2nt.items():
    gene_symbol = nt.Symbol.upper()
    symbol2id[gene_symbol] = gene_id

    for alias in nt.Aliases.split(', '):
        symbol2id[alias.upper()] = gene_id

marker_gene_table_path = os.path.join(
    output_dir, f'{marker_gene_key}_marker_genes.csv')
marker_gene_table = pd.read_csv(marker_gene_table_path, index_col=0)
go_tables = {}

for cluster in np.unique(marker_gene_table['group']):
    # get gene ids from gene symbols
    gene_symbols = marker_gene_table['names'][
        (marker_gene_table['group'] == cluster)
        & (marker_gene_table[filter_col] < marker_gene_max_pval)].tolist()
    gene_ids = [symbol2id[s] for s in gene_symbols]
    go_tables[cluster] = pd.DataFrame(
        columns=['go_id', 'term', 'p_val_adj', 'gene_count', 'genes'])
    go_tables[cluster] = go_tables[cluster].astype({'gene_count': int})

    # run GO study
    if len(gene_ids) > 0:
        go_records = go_study.run_study(gene_ids)

        # collect all significant GO terms
        for gr in go_records:
            if gr.p_fdr_bh < go_max_pval:
                go_genes = [GeneID2nt[i].Symbol.upper() for i in gr.study_items]
                row = {'go_id': gr.goterm.id, 'term': gr.name,
                    'p_val_adj': gr.p_fdr_bh, 'gene_count': gr.study_count,
                    'genes': '.'.join(go_genes)}
                go_tables[cluster] = go_tables[cluster].append(
                    row, ignore_index=True)

        # sort GO terms by adjusted p-value
        go_tables[cluster].sort_values('p_val_adj', ascending=False)

    # export GO table
    go_table_path = os.path.join(
        output_dir, f'go_terms_{marker_gene_key}_{cluster}_{filter_col}.csv')
    go_tables[cluster].to_csv(go_table_path)

# %%
# plot trajectories
reordered_session_indices = np.argsort(adata.obs[cluster_key])

plt.figure(figsize=(4, 6), dpi=300)
_ = sns.heatmap(y_sessions[reordered_session_indices, :], xticklabels=False,
                yticklabels=np.sort(adata.obs[cluster_key]))
plt.xlabel('Time')
plt.ylabel('Ca2+ response')
plt.tight_layout()
figure_path = os.path.join(output_dir, f'{cluster_key}_trajectories.pdf')
plt.savefig(figure_path)
plt.close()

# %%
# plot peaks of trajectories
adata.obs['peak'] = np.amax(y_sessions, axis=1)
with plt.rc_context({"figure.figsize": (4, 4), "figure.dpi": (300)}):
    sc.pl.pca(adata, color='peak', use_raw=False, show=False,
              save='_trajectory_peaks.pdf')

# %%
adata.write(
    os.path.join(output_dir, f'leiden_{leiden_resolution:.2f}_adata.h5ad'))

# %%
