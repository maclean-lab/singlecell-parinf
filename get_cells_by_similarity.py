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

def plot_similarity(similarity_matrix):
        plt.clf()
        plt.figure(figsize=(16, 16))
        plt.imshow(similarity_matrix, cmap="gray")
        plt.colorbar()
        plt.savefig("cell_similarity.png")
        plt.close()

def get_cells_greedy(similarity_matrix):
    num_cells = similarity_matrix.shape[0]

    for i in range(1, num_cells):
        cells = np.zeros(num_cells, dtype=np.int)
        cell_set = set([0])

        sorted_cells = np.argsort(similarity_matrix[cells[i - 1], :])[::-1]
        cells[i] = next(c for c in sorted_cells if c not in cell_set)
        cell_set.add(cells[i])

    cells = pd.Series(cells)
    cells.to_csv("cells_by_similarity.txt", header=False, index=False)

def get_cells_bfs(similarity_matrix, root=0, threshold=0.0):
    curr_level = collections.deque()
    curr_level.append(root)
    next_level = collections.deque()

    num_cells = similarity_matrix.shape[0]
    visited = np.full(num_cells, False, dtype=bool)
    ordered_cells = np.zeros(num_cells, dtype=int)
    parents = np.zeros(num_cells, dtype=int)

    i = 0  # order of cells, not index of cells
    visited[root] = True
    ordered_cells[i] = root
    parents[root] = -1
    while len(curr_level) > 0:
        # get a cell at current BFS level
        cell = curr_level.popleft()

        # add unvisited neighbors to the next level
        for neighbor in range(num_cells):
            if neighbor != cell and not visited[neighbor] and similarity_matrix[cell, neighbor] > threshold:
                visited[neighbor] = True
                i += 1
                ordered_cells[i] = neighbor
                parents[i] = cell
                next_level.append(neighbor)

        # go to next level if all cells in current level have been visited
        if len(curr_level) == 0:
            curr_level = next_level
            next_level = collections.deque()

    print(np.all(visited))
    print(len(set(ordered_cells)) == num_cells)
    bfs_result = pd.DataFrame({"Cell": ordered_cells, "Parent": parents})
    bfs_result.to_csv("cells_by_similarity_bfs.txt", sep="\t", index=False)

if __name__ == "__main__":
    main()
