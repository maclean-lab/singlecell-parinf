#!/usr/bin/env python
import sys
import collections
from argparse import ArgumentParser
import random
import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt

def main():
    args = get_args()
    method = args.method
    root_cell = args.root_cell
    stochastic = args.stochastic
    min_similarity = args.min_similarity
    min_peak = args.min_peak
    output_file = args.output

    if method == 'random':
        get_cells_random(output_file, root=root_cell, min_peak=min_peak)
        return

    # load similarity matrix
    soptsc_vars = scipy.io.loadmat(
        '../../result/SoptSC/SoptSC_feature_100/workspace.mat')
    similarity_matrix = soptsc_vars['W']

    if method == 'plot':
        plot_similarity(similarity_matrix, output_file)
    elif method == 'greedy':
        get_cells_greedy(similarity_matrix, output_file, root=root_cell)
    elif method == 'bfs':
        get_cells_bfs(similarity_matrix, output_file, root=root_cell,
                      min_similarity=min_similarity)
    elif method == 'dfs':
        get_cells_dfs(similarity_matrix, output_file, root=root_cell,
                      stochastic=stochastic, min_similarity=min_similarity,
                      min_peak=min_peak)

def plot_similarity(similarity_matrix, output_file):
    plt.clf()
    plt.figure(figsize=(16, 16))
    plt.imshow(similarity_matrix, cmap='gray')
    plt.colorbar()
    plt.savefig(output_file)
    plt.close()

def get_cells_random(output_file, root=0, min_peak=0.0):
    """get cell list by random shuffle"""
    cells = select_cells(min_peak=min_peak)
    cells.remove(root)
    cells = np.array(list(cells))

    rng = np.random.default_rng()
    rng.shuffle(cells)
    cells = np.insert(cells, 0, root)
    cells = pd.DataFrame(cells, columns=['Cell'])
    cells.to_csv(output_file, sep='\t', index=False)

def get_cells_greedy(similarity_matrix, output_file, root=0, verbose=False):
    num_cells = similarity_matrix.shape[0]
    visited = np.full(num_cells, False, dtype=bool)
    ordered_cells = np.zeros(num_cells, dtype=np.int)
    num_unsimilars = 0
    visited[root] = True

    for i in range(1, num_cells):
        sorted_cells = np.argsort(
            similarity_matrix[ordered_cells[i - 1], :]
        )[::-1]
        ordered_cells[i] = next(c for c in sorted_cells if not visited[c])
        visited[ordered_cells[i]] = True

        if similarity_matrix[ordered_cells[i - 1], ordered_cells[i]] == 0:
            num_unsimilars += 1
            if verbose:
                print(f'Warning: the {i}-th cell is added with 0 similarity')

    if verbose:
        print(f'There are {num_unsimilars} cells that are not similar '
              + 'to their predecessors')

    ordered_cells = pd.Series(ordered_cells)
    ordered_cells.to_csv(output_file, header=False, index=False)

def get_cells_bfs(similarity_matrix, output_file, root=0, min_similarity=0.0):
    """get cell ordering using breadth-first search"""
    # initialize BFS
    num_cells = similarity_matrix.shape[0]
    curr_level = collections.deque()
    next_level = collections.deque()
    visited = np.full(num_cells, False, dtype=bool)
    ordered_cells = np.zeros(num_cells, dtype=int)
    parents = np.zeros(num_cells, dtype=int)

    # run BFS
    i = 0  # order of cell in BFS, not index of cell
    curr_level.append(root)
    visited[root] = True
    ordered_cells[i] = root
    parents[root] = -1
    while len(curr_level) > 0:
        # get a cell at current BFS level
        cell = curr_level.popleft()

        # add unvisited neighbors to the next level
        for neighbor in range(num_cells):
            if (neighbor != cell and not visited[neighbor]
                    and similarity_matrix[cell, neighbor] > min_similarity):
                next_level.append(neighbor)
                visited[neighbor] = True

                i += 1
                ordered_cells[i] = neighbor
                parents[i] = cell

        # go to next level if all cells in current level have been visited
        if len(curr_level) == 0:
            curr_level = next_level
            next_level = collections.deque()

    bfs_result = pd.DataFrame({'Cell': ordered_cells, 'Parent': parents})
    bfs_result.to_csv(output_file, sep='\t', index=False)

def get_cells_dfs(similarity_matrix, output_file, root=0, stochastic=False,
                  min_similarity=0.0, min_peak=0.0):
    """get cell ordering using depth-first search"""
    # initialize DFS
    unvisited_set = select_cells(min_peak=min_peak)
    num_cells = len(unvisited_set)
    dfs_stack = collections.deque()
    parent_stack = collections.deque()
    ordered_cells = np.zeros(num_cells, dtype=int)  # indexed by order of visit
    parents = np.zeros(num_cells, dtype=int)        # indexed by order of visit
    num_children = np.zeros(similarity_matrix.shape[0], dtype=int) # indexed by
                                                                   # cell ID

    # run DFS
    i = 0  # order of cell in DFS, not index of cell
    dfs_stack.append(root)
    parent_stack.append(-1)
    while len(dfs_stack) > 0:
        # get a cell from DFS stack
        cell = dfs_stack.pop()
        parent = parent_stack.pop()

        # add cell to ordering if unvisited
        if cell in unvisited_set:
            unvisited_set.remove(cell)
            ordered_cells[i] = cell
            parents[i] = parent
            if parent > -1:
                num_children[parent] += 1
            i += 1

            # find unvisited neighbors
            unvisited_neighbors = []
            for neighbor in unvisited_set:
                if similarity_matrix[cell, neighbor] > min_similarity:
                    unvisited_neighbors.append(neighbor)

            # shuffle unvisited neighbors if specified
            if stochastic:
                random.shuffle(unvisited_neighbors)

            # add unvisited neighbors to the stack
            for neighbor in unvisited_neighbors:
                dfs_stack.append(neighbor)
                parent_stack.append(cell)

    if i + 1 < num_cells:
        print(f'Warning: {num_cells - i - 1} cell(s) not visited')

    # # find cells with more than 1 child
    # print('Cells with more than 1 child:')
    # print('order\tcell\tchildren')
    # for i, cell in enumerate(ordered_cells):
    #     if num_children[cell] > 1:
    #         children = np.squeeze(np.argwhere(parents == cell))
    #         print(f'{i}\t{cell}\t{children}')

    dfs_result = pd.DataFrame({'Cell': ordered_cells, 'Parent': parents})
    dfs_result.to_csv(output_file, sep='\t', index=False)

def select_cells(min_peak=0.0):
    y = np.loadtxt('canorm_tracjectories.csv', delimiter=',')
    selected = np.where(np.max(y, axis=1) >= min_peak)
    selected = set(selected[0])

    return selected

def get_args():
    """parse command line arguments"""
    arg_parser = ArgumentParser(
        description='Create a list of cells')
    arg_parser.add_argument('--method', type=str, required=True)
    arg_parser.add_argument('--root_cell', type=int, required=True)
    arg_parser.add_argument('--stochastic', default=False, action='store_true')
    arg_parser.add_argument('--min_similarity', type=float, default=0.0)
    arg_parser.add_argument('--min_peak', type=float, default=0.0)
    arg_parser.add_argument('--output', type=str, required=True)

    return arg_parser.parse_args()

if __name__ == '__main__':
    main()
