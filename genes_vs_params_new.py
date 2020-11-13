# %%
import numpy as np
import pandas as pd
from stan_helpers import StanMultiSessionAnalyzer

# %%
root_cell_id, first_cell, last_cell = 5106, 5121, 5120
# stan_run_suffix = '-1'
# stan_run_suffix = '-2'
# stan_run_suffix = '-simple-prior'
# stan_run_suffix = '-const-eta1'
stan_run_suffix = '-const-Be'

# root_cell_id, first_cell, last_cell = 5085, 4982, 4553
# stan_run_suffix = '-1'

all_param_names = ['sigma', 'KonATP', 'L', 'Katp', 'KoffPLC', 'Vplc', 'Kip3',
                   'KoffIP3', 'a', 'dinh', 'Ke', 'Be', 'd1', 'd5', 'epr',
                   'eta1', 'eta2', 'eta3', 'c0', 'k3']
# param_mask = '0111111111111111111'
# param_mask = '0111111111111101111'  # for const eta1
param_mask = '0111111111011111111'  # for const Be
param_names = [all_param_names[i + 1] for i, mask in enumerate(param_mask)
               if mask == "1"]
param_names = ["sigma"] + param_names
num_params = len(param_names)

cell_list_path = f'cell_lists/dfs_feature_100_root_{root_cell_id}_0.000_1.8.txt'
cell_list = pd.read_csv(cell_list_path, sep='\t')
first_cell_order = np.where(cell_list['Cell'] == first_cell)[0][0]
last_cell_order = np.where(cell_list['Cell'] == last_cell)[0][0]
cell_list = cell_list.iloc[first_cell_order:last_cell_order, :]

session_list = [str(c) for c in cell_list['Cell']]
session_dirs = [f'cell-{c:04d}' for c in cell_list['Cell']]
result_root = f'../../result/stan-calcium-model-100-root-{root_cell_id}' \
    + f'{stan_run_suffix}'

# %%
analyzer = StanMultiSessionAnalyzer(session_list, result_root, session_dirs,
                                    param_names=param_names)

# %%
analyzer.get_sample_means()
# analyzer.filter_sessions(z_score_max=3.0)

# %%
analyzer.get_parameter_correlations(plot=True)

# %%
analyzer.plot_parameter_violin()

# %%
analyzer.plot_parameter_ribbon()

# %%
