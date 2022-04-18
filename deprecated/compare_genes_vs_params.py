# %%
import os.path
import pandas as pd

# %%
run_1_dir = os.path.join('../../result/stan-calcium-model-100-root-5106-1')
gvp_sorted_1 = pd.read_csv(
    os.path.join(run_1_dir, 'genes-vs-params', 'genes-vs-params',
                 'pearson_corrs_sorted.csv'),
    index_col=0)

run_2_dir = os.path.join('../../result/stan-calcium-model-100-root-5085-1')
gvp_sorted_2 = pd.read_csv(
    os.path.join(run_2_dir, 'genes-vs-params', 'genes-vs-params',
                 'pearson_corrs_sorted.csv'),
    index_col=0)

# %%
corr_min = 0.2
gvp_high_corrs_1 = gvp_sorted_1.loc[
    gvp_sorted_1['Correlation'].abs() > corr_min]
gvp_high_corrs_2 = gvp_sorted_2.loc[
    gvp_sorted_2['Correlation'].abs() > corr_min]

# %%
high_corr_pairs_1 = set()
for row in gvp_high_corrs_1.index:
    gene = gvp_high_corrs_1.loc[row, 'Gene']
    param = gvp_high_corrs_1.loc[row, 'Parameter']
    high_corr_pairs_1.add((gene, param))

high_corr_pairs_2 = set()
for row in gvp_high_corrs_2.index:
    gene = gvp_high_corrs_2.loc[row, 'Gene']
    param = gvp_high_corrs_2.loc[row, 'Parameter']
    high_corr_pairs_2.add((gene, param))

# %%
common_pairs = high_corr_pairs_1 & high_corr_pairs_2
for gene, param in sorted(list(common_pairs)):
    print(f'{gene:10}{param}')

# %%
