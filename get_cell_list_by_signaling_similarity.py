# %%
# generate a list cell such that two consecutive cells have similar Ca2+
# response
import os.path
import collections
import itertools
import json

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

from stan_helpers import load_trajectories

# %%
# stan_runs = ['3']
stan_runs = ['const-Be-eta1']
# stan_runs = ['const-Be-eta1-mixed-1']
# stan_runs = [f'const-Be-eta1-mixed-{i}' for i in range(5)]
# stan_runs = ['const-Be-eta1-random-1']
# stan_runs = [f'const-Be-eta1-random-{i}' for i in range(1, 7)]
list_ranges = [(1, 500)]
# list_ranges = [(1, 100)]
# list_ranges = [(1, 100), (1, 100), (1, 100), (1, 100), (1, 100)]
# list_ranges = [(1, 372)]
# list_ranges = [(1, 571), (1, 372), (1, 359), (1, 341), (1, 335), (1, 370)]

cluster_method = 'k_means'
num_clusters = 3
cluster_key = f'{cluster_method}_{num_clusters}'

# get cell list
with open('stan_run_meta.json', 'r') as f:
    stan_run_meta = json.load(f)
session_list = []
for run, lr in zip(stan_runs, list_ranges):
    cell_list_path = os.path.join('cell_lists',
                                  stan_run_meta[run]['cell_list'])
    run_cell_list = pd.read_csv(cell_list_path, sep='\t')
    cell_list = run_cell_list.iloc[lr[0]:lr[1] + 1, :]
    root_cell_dir = os.path.join('../../result',
                                 stan_run_meta[run]['root_cell_dir'])
    root_cell_id = int(root_cell_dir[-4:])
    session_list.extend([str(c) for c in cell_list['Cell']])

session_list_int = [int(s) for s in session_list]
num_cells = len(session_list)

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
output_dir = os.path.join(output_root, 'trajectory-clustering')
sc.settings.figdir = output_dir

# change font settings
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['font.size'] = 16

# %%
# load expression data and preprocess
print('Loading gene expression...')
adata = sc.read_csv('vol_adjusted_genes.csv')
# adata = adata[session_list, :]
adata.raw = adata
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.scale(adata)

# %%
# compute signaling similarity between cells
signaling_dists = []
signaling_dist_mat = np.zeros((num_cells, num_cells))

for i, j in itertools.combinations(range(num_cells), 2):
    sd = np.linalg.norm(y_sessions[i, :] - y_sessions[j, :])
    if sd == np.nan:
        print(i, j)
    signaling_dists.append(sd)
    signaling_dist_mat[i, j] = signaling_dist_mat[j, i] = sd

signaling_dists_to_root = np.array(
    [np.linalg.norm(y_all[root_cell_id, :] - y_sessions[i, :])
     for i in range(num_cells)]
)

# %%
# NOTE: do not run the following code on all 500 cells. It takes more than 50
# seconds to get results for 20 cells. Since the algorithm's complexity is
# O(n^2 * 2^n), running on all 500 cells may take years.

# make a list of cells such that the sum of distances between consecutive cells
# is minimal. the first cell is the root cell
# it is equivalent to the traveling salesman problem (without returning to the
# first cell) and we use the Held-Karp algorithm
min_partial_dists = {} # key: (bits_of_traversed_cells, last_cell_idx)
parent_indices = {} # key: (bits_of_traversed_cells, last_cell_idx)

# initialize distances to the root cell
for i in range(num_cells):
    min_partial_dists[(1 << i, i)] = signaling_dists_to_root[i]
    parent_indices[(1 << i, i)] = 's'

# extend paths from root cell
for path_len in range(2, num_cells + 1):
    # iterate over all subset of cells with current path length
    for cell_set in itertools.combinations(range(num_cells), path_len):
        cell_bits = 0
        for i in cell_set:
            cell_bits |= 1 << i

        # get shortest path ending at each cell in the subset
        for i in cell_set:
            min_dist = np.inf
            min_parent = None
            prev_cell_bits = cell_bits & ~(1 << i)

            for j in cell_set:
                if i == j:
                    continue

                curr_dist = min_partial_dists[(prev_cell_bits, j)] + \
                    signaling_dist_mat[i, j]
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    min_parent = j

            min_partial_dists[(cell_bits, i)] = min_dist
            parent_indices[(cell_bits, i)] = min_parent

# find the last cell in the shortest path
min_total_dist = np.inf
shortest_path = [None]
curr_cell_bits = (1 << num_cells) - 1
for i in range(num_cells):
    if min_partial_dists[(curr_cell_bits, i)] < min_total_dist:
        min_total_dist = min_partial_dists[(curr_cell_bits, i)]
        shortest_path[0] = i

# reconstruct path of shortest path
while len(shortest_path) < num_cells:
    next_cell_idx = parent_indices[(curr_cell_bits, shortest_path[-1])]
    curr_cell_bits &= ~(1 << shortest_path[-1])
    shortest_path.append(next_cell_idx)

shortest_path = list(reversed(shortest_path))

# %%
# get cell list using depth-first search
# the generated list is a tree with very few branches near the bottom
cell_dist_upper_bound = 5.0

# initialize DFS
unvisited_cell_indices = set(range(num_cells))
visiting_stack = collections.deque()  # cell indices of cells being visited
parent_stack = collections.deque()  # cell indices of parent cells
cell_list = [root_cell_id]  # cell IDs of visited cells
parent_list = [-1]  # cell IDs of parent cells
num_children = np.zeros(num_cells, dtype=int)

# add all cells similar to root cell to visiting stack
for i in range(num_cells):
    if signaling_dists_to_root[i] <= cell_dist_upper_bound:
        visiting_stack.append(i)
        parent_stack.append(-1)

# visit all other cells by DFS
while len(visiting_stack) > 0:
    # get a visiting cell
    cell_idx = visiting_stack.pop()
    parent_idx = parent_stack.pop()

    # add cell to cell list if unvisited
    if cell_idx in unvisited_cell_indices:
        unvisited_cell_indices.remove(cell_idx)
        cell_list.append(session_list_int[cell_idx])
        if parent_idx == -1:
            parent_list.append(root_cell_id)
        else:
            parent_list.append(session_list_int[parent_idx])
        num_children[parent_idx] += 1

        # add unvisited neighbors to visiting stack
        for idx in unvisited_cell_indices:
            if signaling_dist_mat[cell_idx, idx] <= cell_dist_upper_bound:
                visiting_stack.append(idx)
                parent_stack.append(cell_idx)

# save cell list
output_path = os.path.join('cell_lists', 'signaling_similarity.txt')
full_cell_list = pd.DataFrame({'Cell': cell_list, 'Parent': parent_list})
full_cell_list.to_csv(output_path, sep='\t', index=False)

# %%
# compare distances between consecutive cells in the list and between all pairs
# of cells
cell_list_dists = [np.linalg.norm(y_all[i, :] - y_all[j, :])
                   for i, j in zip(cell_list[1:], parent_list[1:])]
plt.hist([cell_list_dists, signaling_dists], density=True)

# %%
# plot gene expression of cells as ordered by the generated cell list
