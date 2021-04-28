# %%
import os
import os.path
import itertools
import numpy as np
import pandas as pd

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
