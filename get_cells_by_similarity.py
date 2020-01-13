#!/usr/bin/env python
import sys
import collections
import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt

def main():
    args = sys.argv[1:]

    # load similarity matrix
    soptsc_vars = scipy.io.loadmat("../../result/SoptSC/workspace.mat")
    similarity_matrix = soptsc_vars["W"]

    # plot similarity matrix
    if "plot" in args:
        plot_similarity(similarity_matrix)

    if "greedy" in args:
        get_cells_greedy(similarity_matrix)

    if "bfs" in args:
        get_cells_bfs(similarity_matrix)

    if "dfs" in args:
        get_cells_dfs(similarity_matrix)

def plot_similarity(similarity_matrix):
        plt.clf()
        plt.figure(figsize=(16, 16))
        plt.imshow(similarity_matrix, cmap="gray")
        plt.colorbar()
        plt.savefig("cell_similarity.png")
        plt.close()

def get_cells_greedy(similarity_matrix, verbose=False):
    num_cells = similarity_matrix.shape[0]
    cell_set = set([0])
    num_unsimilars = 0

    for i in range(1, num_cells):
        cells = np.zeros(num_cells, dtype=np.int)

        sorted_cells = np.argsort(similarity_matrix[cells[i - 1], :])[::-1]
        cells[i] = next(c for c in sorted_cells if c not in cell_set)
        cell_set.add(cells[i])

        if similarity_matrix[cells[i - 1], cells[i]] == 0:
            num_unsimilars += 1
            if verbose:
                print("Warning: the {}-th cell is added with 0 ".format(i)
                      + "similarity")

    if verbose:
        print("There are {} cells that are not similar ".format(num_unsimilars)
            + "to their predecessors")

    cells = pd.Series(cells)
    cells.to_csv("cells_by_similarity.txt", header=False, index=False)

def get_cells_bfs(similarity_matrix, root=0, threshold=0.0):
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
                    and similarity_matrix[cell, neighbor] > threshold):
                next_level.append(neighbor)
                visited[neighbor] = True

                i += 1
                ordered_cells[i] = neighbor
                parents[i] = cell

        # go to next level if all cells in current level have been visited
        if len(curr_level) == 0:
            curr_level = next_level
            next_level = collections.deque()

    bfs_result = pd.DataFrame({"Cell": ordered_cells, "Parent": parents})
    bfs_result.to_csv("cells_by_similarity_bfs.txt", sep="\t", index=False)

def get_cells_dfs(similarity_matrix, root=0, threshold=0.0):
    """get cell ordering using depth-first search"""
    # initialize DFS
    num_cells = similarity_matrix.shape[0]
    dfs_stack = collections.deque()
    parent_stack = collections.deque()
    visited = np.full(num_cells, False, dtype=bool)
    ordered_cells = np.zeros(num_cells, dtype=int)
    parents = np.zeros(num_cells, dtype=int)

    # run DFS
    i = 0  # order of cell in DFS, not index of cell
    dfs_stack.append(root)
    parent_stack.append(-1)
    while len(dfs_stack) > 0:
        # get a cell from DFS stack
        cell = dfs_stack.pop()
        parent = parent_stack.pop()

        # add cell to ordering if unvisited
        if not visited[cell]:
            visited[cell] = True
            ordered_cells[i] = cell
            parents[i] = parent
            i += 1

            # add unvisited neighbors to the stack
            for neighbor in range(num_cells):
                if (neighbor != cell and not visited[neighbor]
                        and similarity_matrix[cell, neighbor] > threshold):
                    dfs_stack.append(neighbor)
                    parent_stack.append(cell)

    dfs_result = pd.DataFrame({"Cell": ordered_cells, "Parent": parents})
    dfs_result.to_csv("cells_by_similarity_dfs.txt", sep="\t", index=False)

if __name__ == "__main__":
    main()
