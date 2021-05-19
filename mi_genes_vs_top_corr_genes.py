# %%
import os.path
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

# %%
# load gene-gene MI data
gene_gene_mi_matrix = pd.read_csv('../../result/gene_gene_MI.csv', index_col=0)
gene_gene_mi_long = pd.DataFrame(columns=['gene_1', 'gene_2', 'MI'])
for g1, g2 in itertools.combinations(gene_gene_mi_matrix.index, 2):
    row = {'gene_1': g1, 'gene_2': g2, 'MI': gene_gene_mi_matrix.loc[g1, g2]}
    gene_gene_mi_long = gene_gene_mi_long.append(row, ignore_index=True)
gene_gene_mi_long.sort_values('MI', axis=0, ascending=False, inplace=True,
                              ignore_index=True)

# get genes with highest MI
num_top_genes = 30
high_mi_genes = set(gene_gene_mi_long.loc[:num_top_genes, 'gene_1'])
high_mi_genes |= set(gene_gene_mi_long.loc[:num_top_genes, 'gene_2'])

# %%
# load gene-param pairs
run_dir  = '../../result/stan-calcium-model-100-root-5106-const-Be-eta1'
gene_param_corr_path = os.path.join(
    run_dir, 'multi-sample-analysis-0001-0500', 'genes-vs-params',
    'pearson_corrs_sorted.csv')
gene_param_corrs = pd.read_csv(gene_param_corr_path, index_col=0)

# get genes with highest correlations
high_corr_genes = set(gene_param_corrs.loc[:num_top_genes, 'Gene'])

# %%
print('Top genes from both analyses:')
overlapping_genes = high_mi_genes & high_corr_genes
print(overlapping_genes)

print('Top genes only in MI analysis:')
high_mi_only_genes = high_mi_genes - high_corr_genes
print(high_mi_only_genes)

print('Top genes only in correlation analysis:')
high_corr_only_genes = high_corr_genes - high_mi_genes
print(high_corr_only_genes)

# %%
# make Venn diagram
plt.figure(figsize=(6, 4), dpi=300)
v_handle = venn2([high_mi_genes, high_corr_genes],
                 set_labels=['MI gene-gene pairs','Gene-param pairs'])

# change text on each patch to corresponding genes
v_handle.get_label_by_id('10').set_text('\n'.join(high_mi_only_genes))
v_handle.get_label_by_id('01').set_text('\n'.join(high_corr_only_genes))
v_handle.get_label_by_id('11').set_text('\n'.join(overlapping_genes))

figure_path = os.path.join(run_dir, 'multi-sample-analysis-0001-0500',
                           'high_MI_vs_high_corr.pdf')
plt.savefig(figure_path)
plt.close()

# %%