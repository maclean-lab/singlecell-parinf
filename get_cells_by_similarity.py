import numpy as np
import scipy.io
import pandas as pd

def main():
    # load similarity matrix
    soptsc_vars = scipy.io.loadmat("../../result/SoptSC/workspace.mat")
    similarity_matrix = soptsc_vars["W"]

    num_cells = similarity_matrix.shape[0]
    cells = np.zeros(num_cells, dtype=np.int)
    cell_set = set([0])

    for i in range(1, num_cells):
        sorted_cells = np.argsort(similarity_matrix[cells[i - 1], :])[::-1]
        cells[i] = next(c for c in sorted_cells if c not in cell_set)
        cell_set.add(cells[i])

    cells = pd.Series(cells)
    cells.to_csv("cells_by_similarity.txt", header=False, index=False)

if __name__ == "__main__":
    main()
