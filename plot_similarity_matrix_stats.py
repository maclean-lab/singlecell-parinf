#!/usr/bin/env python
import os.path
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

result_dir = "../../result/SoptSC_similarity_matrix_stats"

def main():
    # load similarity matrix
    soptsc_vars = scipy.io.loadmat("../../result/SoptSC/workspace.mat")
    similarity_matrix = soptsc_vars["W"]
    plot_neighbor_stats(similarity_matrix)
    plot_similarity_scores(similarity_matrix)

def plot_neighbor_stats(similarity_matrix):
    num_cells = similarity_matrix.shape[0]
    num_neighbors = [np.sum(similarity_matrix[i, :] > 0)
                     for i in range(num_cells)]

    print("Average number of neighbors: {}".format(np.mean(num_neighbors)))

    plt.clf()
    plt.hist(num_neighbors, bins=20)
    plt.savefig(os.path.join(result_dir, "num_neighbor_histogram.png"))
    plt.close()

def plot_similarity_scores(similarity_matrix):
    similarity_matrix = similarity_matrix.flatten()

    print("Average similarity score: {}".format(np.mean(similarity_matrix)))

    plt.clf()
    plt.hist(similarity_matrix, bins=40)
    plt.savefig(os.path.join(result_dir, "similarity_score_histogram.png"))
    plt.close()

if __name__ == "__main__":
    main()
