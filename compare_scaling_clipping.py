# %%
import os.path
import matplotlib
import matplotlib.pyplot as plt

# change font settings
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['font.size'] = 16

# %%
first_cell_order = 1
last_cell_order = 36

output_root = '../../result'
cell_chains = ['3', '3-1.0', '3-2.0']
analyzer_dirs = [
    os.path.join(
        output_root, f'stan-calcium-model-100-root-5106-{r}',
        f'multi-sample-analysis-{first_cell_order:04d}-{last_cell_order:04d}')
    for r in cell_chains]
output_dir = os.path.join(
    output_root, f'stan-calcium-model-100-root-5106-comparison-scaling',
    f'cell-{first_cell_order:04d}-{last_cell_order:04d}')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
num_runs = len(analyzer_dirs)

# %%
