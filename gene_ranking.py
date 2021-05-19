# %%
import os
import os.path
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

# %%
# load genes from gene-param correlation
run_dir = 'stan-calcium-model-100-root-5106-const-Be-eta1'
first_cell_order, last_cell_order = 1, 500
gene_param_path = os.path.join(
    '../../result', run_dir,
    f'multi-sample-analysis-{first_cell_order:04d}-{last_cell_order:04d}',
    'genes-vs-params', 'pearson_corrs_sorted.csv')
gene_param_corrs = pd.read_csv(gene_param_path, index_col=0)

# load MI data
mi_matrix = pd.read_csv('../../result/gene_gene_MI.csv', index_col=0)
mi_long = pd.DataFrame(columns=['gene_1', 'gene_2', 'MI'])
for g1, g2 in itertools.combinations(mi_matrix.index, 2):
    row = {'gene_1': g1, 'gene_2': g2, 'MI': mi_matrix.loc[g1, g2]}
    mi_long = mi_long.append(row, ignore_index=True)
mi_long.sort_values('MI', axis=0, ascending=False, inplace=True,
                    ignore_index=True)
mi_genes = mi_matrix.index.tolist()

# %%
# ranking 1: top correlation
ranking_1 = gene_param_corrs['Gene'].unique()
ranking_1 = [gene for gene in ranking_1 if gene in mi_genes]

# %%
# ranking 2: take genes with |corr|>0.2, count # of occurrences of each gene
# order by the count. tie breaker: |top correlation value|
corr_threshold = 0.2
num_occurrences = {}
max_corrs = {}
significant_corrs = gene_param_corrs.loc[
    gene_param_corrs['Correlation'].abs() > corr_threshold, :]
for _, row in significant_corrs.iterrows():
    gene = row['Gene']

    if gene in mi_genes:
        if gene in num_occurrences:
            num_occurrences[gene] += 1
        else:
            num_occurrences[gene] = 1
            max_corrs[gene] = row['Correlation']

ranking_2 = pd.DataFrame(data={'Count': num_occurrences, 'MaxCorr': max_corrs})
ranking_2.sort_values(['Count', 'MaxCorr'], ascending=False, inplace=True,
                      key=lambda x: np.abs(x))

# %%
# ranking 3: for the first X rows (or all rows), use X - row_idx as weight,
# compute total weighted correlation for each gene
corr_threshold = 0.2
significant_corrs = gene_param_corrs.loc[
    gene_param_corrs['Correlation'].abs() > corr_threshold, :]
num_significant_rows = significant_corrs.shape[0]
ranking_3 = {}
for i, row in significant_corrs.iterrows():
    gene = row['Gene']

    if gene in mi_genes:
        weight = num_significant_rows - i
        corr = row['Correlation']
        if gene in ranking_3:
            ranking_3[gene] += weight * np.abs(corr)
        else:
            ranking_3[gene] = weight * np.abs(corr)

ranking_3 = pd.Series(ranking_3)
ranking_3.sort_values(ascending=False, inplace=True)

# %%
ranking_length = np.amin(
    [len(ranking_1), ranking_2.shape[0], ranking_2.shape[0]])
ranking_all = pd.DataFrame(data={'Ranking_1': ranking_1[:ranking_length],
                                 'Ranking_2': ranking_2.index[:ranking_length],
                                 'Ranking_3': ranking_3.index[:ranking_length]})

# %%
# load gene-Ca MI data
gene_ca_mi_data = pd.read_csv(
    '../../result/volume_norm_genes_noisy_ca_mine_1gx314tps_030821_95th%ile_bits.csv',
    index_col=0)
gene_ca_mi_data['mean'] = gene_ca_mi_data.mean(numeric_only=True, axis=1)
gene_ca_mi_data.sort_values('mean', axis=0, ascending=False, inplace=True)
high_mi_genes = set(gene_ca_mi_data.index[:ranking_length])

# %%
# make Venn diagram
for ranking, ranked_genes in ranking_all.items():
    ranked_genes = set(ranked_genes)
    overlapping_genes = ranked_genes & high_mi_genes
    high_mi_only_genes = high_mi_genes - ranked_genes
    high_corr_only_genes = ranked_genes - high_mi_genes

    plt.figure(figsize=(6, 4), dpi=300)
    v_handle = venn2(
        [high_mi_genes, ranked_genes],
        set_labels=['From Gene-Ca MI','From gene-param correlation'])

    # change text on each patch to corresponding genes
    v_handle.get_label_by_id('10').set_text('\n'.join(high_mi_only_genes))
    v_handle.get_label_by_id('01').set_text('\n'.join(high_corr_only_genes))
    v_handle.get_label_by_id('11').set_text('\n'.join(overlapping_genes))

    figure_path = os.path.join('../../result', run_dir,
                               'multi-sample-analysis-0001-0500',
                               f'high_MI_vs_high_corr_{ranking.lower()}.pdf')
    plt.savefig(figure_path)
    plt.close()

# %%
