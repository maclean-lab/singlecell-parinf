# Cell lists
This folder contains ordered lists of cells generated from similarity matrices.

## Specification
### File names
Each cell list is named by the following format:
```
{method}_{soptsc_params}_{min_similarity}_{min_peak}[_{deterministic|stochastic}].txt
```
where
* `method` is the graph traversal method used for generating the list.
* `soptsc_params` shows the parameters used in SoptSC to get similarity matrix,
e.g. number of features.
* `min_similarity` is the minimum similarity score for two cells to be
considered neighbors.
* `min_peak` is the lowest possible peak value of a cell's calcium response
trajectory for the cell to be included in the cell list. In other words, if the
trajectory of a cell is lower than `min_peak` everywhere, it will not be in the
list.
* `deterministic|stochastic` is an optional descriptor, which indicates whether
new nodes are added to the FIFO queue of BFS or stack of DFS in a deterministic
or stochastic manner.

### File content
Each file contains a tab-delimited table.

A table from BFS or DFS has two columns. The first column, `Cell`, is a list of
cells in BFS or DFS order. The second column, `Parent`, holds the parent cells
during traversal.

A table from greedy algorithm has one column of ordered cells with no header.
