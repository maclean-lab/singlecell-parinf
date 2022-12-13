#!/usr/bin/env python
# Plot gene expression by order in cell chain
import argparse
from matplotlib.pyplot import axis, figure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    args = get_args()
    cell_list_path = args.cell_list
    figure_path = args.figure_path
    add_dendrogram = args.add_dendrogram

    # load gene expression
    raw_data = pd.read_csv('vol_adjusted_genes_transpose.txt',
                           sep='\t')
    log_data = np.log1p(raw_data.to_numpy())

    # load cell list
    if cell_list_path:
        cell_list = pd.read_csv(cell_list_path, delimiter="\t", index_col=False)
        log_data = log_data[:, cell_list['Cell']]

    # make plot
    if add_dendrogram:
        sns.clustermap(log_data, standard_scale=1)
    else:
        # normalize log data
        normalized_data = log_data / log_data.sum(axis=0)[np.newaxis, :]

        plt.figure(figsize=(8, 20), dpi=300)
        plt.imshow(normalized_data.T, aspect='auto')
        plt.colorbar()
        plt.tight_layout()

    # save plot
    plt.savefig(figure_path)

def get_args():
    arg_parser = argparse.ArgumentParser(
        description='Plot gene expression by order in cell chain')
    arg_parser.add_argument('--cell_list', type=str, default=None)
    arg_parser.add_argument('--figure_path', type=str, required=True)
    arg_parser.add_argument('--add_dendrogram', action='store_true')

    return arg_parser.parse_args()

if __name__ == '__main__':
    main()
