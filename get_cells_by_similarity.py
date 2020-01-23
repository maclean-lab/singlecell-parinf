#!/usr/bin/env python
import sys
import collections
import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt

def main():
    method = sys.argv[1]
    threshold = 0.0 if len(sys.argv) < 3 else float(sys.argv[2])

    # load similarity matrix
    soptsc_vars = scipy.io.loadmat("../../result/SoptSC/workspace.mat")
    similarity_matrix = soptsc_vars["W"]

    # plot similarity matrix
    if method == "plot":
        plot_similarity(similarity_matrix)
    elif method == "greedy":
        get_cells_greedy(similarity_matrix)
    elif method == "bfs":
        get_cells_bfs(similarity_matrix, threshold=threshold)
    elif method == "dfs":
        get_cells_dfs(similarity_matrix, threshold=threshold)

def plot_similarity(similarity_matrix):
        plt.clf()
        plt.figure(figsize=(16, 16))
        plt.imshow(similarity_matrix, cmap="gray")
        plt.colorbar()
        plt.savefig("cell_similarity.png")
        plt.close()

def get_cells_greedy(similarity_matrix, root=0, verbose=False):
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
                print("Warning: the {}-th cell is added with 0 ".format(i)
                      + "similarity")

    if verbose:
        print("There are {} cells that are not similar ".format(num_unsimilars)
            + "to their predecessors")

    ordered_cells = pd.Series(ordered_cells)
    ordered_cells.to_csv("cells_by_similarity_greedy.txt", header=False,
                         index=False)

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
    num_visited = 0

    # run DFS
    i = 0  # order of cell in DFS, not index of cell
    dfs_stack.append(root)
    parent_stack.append(-1)
    while len(dfs_stack) > 0:
        # get a cell from DFS stack
        cell = dfs_stack.pop()
        parent = parent_stack.pop()
        num_visited += 1

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

    if num_visited < num_cells:
        print("Warning: {} cell(s) not visited".format(num_cells - num_visited))

    dfs_result = pd.DataFrame({"Cell": ordered_cells, "Parent": parents})
    dfs_result.to_csv("cells_by_similarity_dfs.txt", sep="\t", index=False)

if __name__ == "__main__":
    main()
