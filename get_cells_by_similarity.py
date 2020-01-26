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
    soptsc_vars = scipy.io.loadmat("../../result/SoptSC/SoptSC/workspace.mat")
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
    unvisited_set = set(range(num_cells))
    ordered_cells = np.zeros(num_cells, dtype=int)  # indexed by order of visit
    parents = np.zeros(num_cells, dtype=int)        # indexed by cell ID
    num_children = np.zeros(num_cells, dtype=int)   # indexed by cell ID

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

            # add unvisited neighbors to the stack
            for neighbor in unvisited_set:
                if similarity_matrix[cell, neighbor] > threshold:
                    dfs_stack.append(neighbor)
                    parent_stack.append(cell)

    if i + 1 < num_cells:
        print("Warning: {} cell(s) not visited".format(num_cells - i - 1))

    # find cells with more than 1 child
    print("Cells with more than 1 child:")
    print("order\tcell\tchildren")
    for i, cell in enumerate(ordered_cells):
        if num_children[cell] > 1:
            children = np.squeeze(np.argwhere(parents == cell))
            print("{}\t{}\t{}".format(i, cell, children))

    dfs_result = pd.DataFrame({"Cell": ordered_cells, "Parent": parents})
    dfs_result.to_csv("cells_by_similarity_dfs.txt", sep="\t", index=False)

if __name__ == "__main__":
    main()
