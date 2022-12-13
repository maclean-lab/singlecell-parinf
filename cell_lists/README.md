# Cell lists
This folder contains ordered lists of cells generated from similarity matrices.

## Specification
### File names
Each cell list is named by the following format:
```
{method}_{soptsc_params}_root_{cell_id}_{min_similarity}_{min_peak}[_{note}].txt
```
where
* `method` is the graph traversal method used for generating the list.
* `soptsc_params` shows the parameters used in SoptSC to get similarity matrix,
e.g. number of features.
* `cell_id` is the numeric ID of the first cell in the list.
* `min_similarity` is the minimum similarity score for two cells to be
considered neighbors.
* `min_peak` is the lowest possible peak value of a cell's calcium response
trajectory for the cell to be included in the cell list. In other words, if the
trajectory of a cell is lower than `min_peak` everywhere, it will not be in the
list.
* `note` is optional and can be one of the following:
    * `deterministic` or `stochastic` indicates whether new nodes are added to
    the FIFO queue of BFS or stack of DFS in a deterministic or stochastic
    manner.
    * `reversed_{branch_cell_id}_{idx}` is appended for reversed cell lists
    from an existing DFS or BFS list. Lists with the same `branch_cell_id` share
    the same tail starting from the cell with numeric ID `branch_cell_id` but
    may have different heads before that cell.

One exception is `signaling_similarity.txt`, which is a cell list based on
similarity of Ca<sup>2+</sup> response between cells.

### File content
Each file contains a tab-delimited table.

A table from BFS or DFS has two columns. The first column, `Cell`, is a list of
cells in BFS or DFS order. The second column, `Parent`, holds the parent cells
during traversal.

A table from greedy algorithm has one column of ordered cells with no header.
