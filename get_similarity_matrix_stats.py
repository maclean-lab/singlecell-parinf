#!/usr/bin/env python
import os.path
import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

result_dir = "../../result/SoptSC/SoptSC_similarity_matrix_stats"

def main():
    # load similarity matrix
    soptsc_vars = scipy.io.loadmat("../../result/SoptSC/SoptSC/workspace.mat")
    similarity_matrix = soptsc_vars["W"]
    # plot_neighbor_stats(similarity_matrix)
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
    positive_similarity_scores = similarity_matrix[similarity_matrix > 0]
    positive_similarity_scores = positive_similarity_scores.flatten()

    print("Average similarity score: "
          + "{}".format(np.mean(positive_similarity_scores)))

    # make histogram
    plt.clf()
    plt.hist(positive_similarity_scores, bins=20)
    plt.savefig(os.path.join(result_dir, "similarity_score_histogram.png"))
    plt.close()

    plt.clf()
    sns.kdeplot(positive_similarity_scores)
    plt.savefig(os.path.join(result_dir, "similarity_score_kde.png"))
    plt.close()


    # compute percentiles
    percentages = np.arange(0, 100, 5)
    percentiles = np.percentile(positive_similarity_scores, percentages)
    percentile_table = pd.DataFrame({"percentage": percentages,
                                     "percentile": percentiles})
    print(percentile_table)

if __name__ == "__main__":
    main()
