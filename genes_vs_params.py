# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os.path
import itertools
import math
from random import sample
import numpy as np
from pandas.core.arrays.sparse import dtype
import scipy.stats
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData

from stan_helpers import StanSessionAnalyzer

# set flags
use_highly_variable_genes = False

# %%
# load sampled parameters
# get cell list
cell_list_path = 'cell_lists/dfs_feature_100_root_5106_0.000_1.8.txt'
stan_output_root = '../../result/stan-calcium-model-100-root-5106-1'
if use_highly_variable_genes:
    output_dir = os.path.join(stan_output_root, 'hv_genes_vs_params')
else:
    output_dir = os.path.join(stan_output_root, 'genes_vs_params')

cell_list = pd.read_csv(cell_list_path, sep='\t')
first_cell = 5029
last_cell = 4532
first_cell_order = np.where(cell_list['Cell'] == first_cell)[0][0]
last_cell_order = np.where(cell_list['Cell'] == last_cell)[0][0]

param_names = ['sigma', 'L', 'Katp', 'KoffPLC', 'Vplc', 'Kip3', 'KoffIP3',
               'a', 'dinh', 'Ke', 'Be', 'd1', 'd5', 'epr', 'eta1', 'eta2',
               'eta3', 'c0', 'k3']
num_params = len(param_names)

# %%
# load sampled parameters for each cell
sampled_cell_ids = []
sample_means = {p: [] for p in param_names}
for cell_order in range(first_cell_order, last_cell_order + 1):
    # load samples
    cell_id = cell_list.loc[cell_order, 'Cell']
    cell_path = os.path.join(stan_output_root, f'cell-{cell_id:04d}')
    analyzer = StanSessionAnalyzer(cell_path, sample_source='arviz_inf_data',
                                   param_names=param_names)

    # compute sample means for cells with mixed chains (R_hat < 4.0)
    cell_sample_means = analyzer.get_sample_means(rhat_upper_bound=4.0)
    if cell_sample_means:
        sampled_cell_ids.append(cell_id)

        for param in param_names:
            sample_means[param].append(cell_sample_means[param])

num_sampled_cells = len(sampled_cell_ids)
cell_order_list = list(range(num_sampled_cells)) # orders of cells in the chain

# %%
# load gene expression
raw_data = pd.read_csv('../../data/vol_adjusted_genes_transpose.txt', sep='\t')

# perform log normalization
if use_highly_variable_genes:
    # use highly variable genes only
    adata = AnnData(raw_data.T)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)
    gene_symbols = adata.var_names[adata.var['highly_variable']]

    log_data = np.log1p(raw_data.loc[gene_symbols, :].to_numpy())
else:
    # use all genes
    log_data = np.log1p(raw_data.to_numpy())
    gene_symbols = raw_data.index.to_numpy()

# %%
# run PCA on log expression
num_comps = 50
num_top_comps = 10
pca = PCA(n_components=num_comps)
log_data_reduced = pca.fit_transform(log_data.T)
log_data_reduced_abs = np.abs(log_data_reduced)

# %%
# plot variance ratio explained by each component
plt.plot(pca.explained_variance_ratio_, '.')
plt.savefig(os.path.join(output_dir, 'pca_var_ratio.png'))
plt.close()

# %%
# plot sample mean vs top principal components
pc_vs_param_pearson_corrs = pd.DataFrame(
    0, columns=range(num_top_comps), index=param_names)
pc_vs_param_pearson_corr_p_vals = pd.DataFrame(
    0, columns=range(num_top_comps), index=param_names)

for param, comp in itertools.product(param_names, range(num_top_comps)):
    corr, p_val = scipy.stats.pearsonr(
        log_data_reduced_abs[sampled_cell_ids, comp], sample_means[param])
    pc_vs_param_pearson_corrs.loc[param, comp] = corr
    pc_vs_param_pearson_corr_p_vals.loc[param, comp] = p_val

# save Pearson correlations to file
pc_vs_param_pearson_corrs.to_csv(
    os.path.join(output_dir, 'pcs_vs_params', 'pearson_corrs.csv'))
pc_vs_param_pearson_corr_p_vals.to_csv(
    os.path.join(output_dir, 'pcs_vs_params', 'pearson_corrs_p_vals.csv'))

# %%
from matplotlib.backends.backend_pdf import PdfPages

def scatter_multi_plot(X_data, output_path, c=None, num_rows=4, num_cols=2,
                       param_regressors=None, X_poly=None):
    """Make multiple scatter plots of some data vs param means in a PDF"""
    num_subplots_per_page = num_rows * num_cols
    num_plots = num_params
    num_pages = math.ceil(num_plots / num_subplots_per_page)

    with PdfPages(output_path) as pdf:
        # generate each page
        for page in range(num_pages):
            # set page size as US letter
            plt.figure(figsize=(8.5, 11))

            # set number of subplots in current page
            if page == num_pages - 1:
                num_subplots = (num_plots - 1) % num_subplots_per_page + 1
            else:
                num_subplots = num_subplots_per_page

            # make each subplot
            for plot_idx in range(num_subplots):
                plt.subplot(num_rows, num_cols, plot_idx + 1)
                param = param_names[page * num_subplots_per_page + plot_idx]
                if isinstance(c, dict):
                    plt.scatter(X_data, sample_means[param], c=c[param])
                else:
                    plt.scatter(X_data, sample_means[param], c=c)
                plt.title(param)
                if c is not None:
                    plt.colorbar()

                # plot regression line/curve
                if param_regressors:
                    sample_mean_pred = param_regressors[param].predict(X_poly)
                    plt.scatter(X_data, sample_mean_pred)

            plt.tight_layout()
            pdf.savefig()
            plt.close()

# %%
# make scatter plots for components vs params
for comp in range(num_top_comps):
    comp_data = log_data_reduced_abs[sampled_cell_ids, comp]
    comp_param_plot_path = os.path.join(output_dir, 'pcs_vs_params',
                                        f'comp_{comp}_vs_params.pdf')
    scatter_multi_plot(comp_data, comp_param_plot_path, c=cell_order_list)

# %%
# plot sample mean of params vs expression of every gene
for gene_idx, gene in enumerate(gene_symbols):
    gene_scatter_path = os.path.join(output_dir, 'genes_vs_params',
                                     f'{gene}.pdf')
    scatter_multi_plot(log_data[gene_idx, sampled_cell_ids], gene_scatter_path,
                       c=cell_order_list)

# %%
# plot z-scores of sample means
sample_mean_z_scores = {p: scipy.stats.zscore(sample_means[p], ddof=1)
                        for p in param_names}

for gene_idx, gene in enumerate(gene_symbols):
    gene_scatter_path = os.path.join(output_dir, 'genes_vs_params',
                                     f'{gene}_z_scores.pdf')
    scatter_multi_plot(log_data[gene_idx, sampled_cell_ids], gene_scatter_path,
                       c=sample_mean_z_scores)

is_cell_outlier = np.zeros(num_sampled_cells, dtype=bool)
sample_mean_z_score_max = 3.0
for i in range(num_sampled_cells):
    is_cell_outlier[i] = np.any(
        [sample_mean_z_scores[p][i] > sample_mean_z_score_max
         for p in param_names])

# %%
# regression analysis for principal components vs sampled means of params
from sklearn.linear_model import LinearRegression, ElasticNetCV, \
    PoissonRegressor, TweedieRegressor, GammaRegressor, HuberRegressor, \
    RANSACRegressor, TheilSenRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

def run_regression_pcs_vs_params(regressor_name, degree=1):
    '''Fit a regression model principal components vs sampled means of params,
    with features raised to a specified degree'''
    print(f'Running {regressor_name} with degree {degree}')
    output_sub_dir = 'pcs_vs_params_regression'

    r2_scores = pd.DataFrame(0, index=param_names, columns=range(num_top_comps))
    mean_sq_errors = pd.DataFrame(0, index=param_names,
                                  columns=range(num_top_comps))
    for comp in range(num_top_comps):
        param_regressors = {}
        # get component data, of shape num_cells * 1
        X_comp = log_data_reduced_abs[sampled_cell_ids, comp, np.newaxis]
        # generate features from component data
        if degree > 1:
            poly = PolynomialFeatures(degree)
            X = poly.fit_transform(X_comp)
        else:
            X = X_comp

        # perform regression for each param
        for param in param_names:
            regressor = regressor_classes[regressor_name]()
            y = sample_means[param]

            regressor.fit(X, y)
            y_pred = regressor.predict(X)
            param_regressors[param] = regressor

            # compute metrics
            r2_scores.loc[param, comp] = regressor.score(X, y)
            mean_sq_errors.loc[param, comp] = mean_squared_error(y, y_pred)

        # plot regression lines (curves)
        regression_scatter_path = os.path.join(
            output_dir, output_sub_dir,
            f'{r}_degree_{degree}_comp_{comp}_scatter.pdf')
        scatter_multi_plot(X_comp, regression_scatter_path,
                           param_regressors=param_regressors, X_poly=X)

    # save metrics
    r2_scores_path = os.path.join(
        output_dir, output_sub_dir, f'{r}_degree_{degree}_scores.csv')
    r2_scores.to_csv(r2_scores_path, float_format='%.8f')

    mean_sq_errors_path = os.path.join(
        output_dir, output_sub_dir, f'{r}_degree_{degree}_mse.csv')
    mean_sq_errors.to_csv(mean_sq_errors_path, float_format='%.8f')

regressor_classes = {'linear': LinearRegression, 'elastic_net': ElasticNetCV,
                     'poisson': PoissonRegressor, 'normal': TweedieRegressor,
                     'gamma': GammaRegressor, 'huber': HuberRegressor,
                     'ransac': RANSACRegressor, 'theil': TheilSenRegressor}

for r, d in itertools.product(regressor_classes, range(1, 5)):
    run_regression_pcs_vs_params(r, degree=d)

# %%
# get PCA loadings
V = pca.components_.T

# find top genes from loadings
num_top_genes = 10
top_pos_genes = pd.DataFrame('', index=range(num_top_genes),
                             columns=range(num_comps))
top_neg_genes = pd.DataFrame('', index=range(num_top_genes),
                             columns=range(num_comps))
for comp in range(num_comps):
    ranked_gene_indices = np.argsort(V[:, comp])
    top_pos_genes_comp = gene_symbols[ranked_gene_indices[:-num_top_genes-1:-1]]
    top_pos_genes.loc[:, comp] = top_pos_genes_comp
    top_neg_genes_comp = gene_symbols[ranked_gene_indices[:num_top_genes]]
    top_neg_genes.loc[:, comp] = top_neg_genes_comp

    if comp < num_top_comps:
        # plot top positive genes vs sampled means
        for i, gene in enumerate(top_pos_genes_comp):
            gene_idx = ranked_gene_indices[-i]
            gene_log_data = log_data[gene_idx, sampled_cell_ids]
            gene_scatter_path = os.path.join(
                output_dir, 'pca_top_genes',
                f'comp_{comp}_top_pos_{i:02d}_{gene}_vs_params.pdf')
            scatter_multi_plot(gene_log_data, gene_scatter_path)

        # plot top negative genes vs sampled means
        for i, gene in enumerate(top_neg_genes_comp):
            gene_idx = ranked_gene_indices[i]
            gene_log_data = log_data[gene_idx, sampled_cell_ids]
            gene_scatter_path = os.path.join(
                output_dir, 'pca_top_genes',
                f'comp_{comp}_top_neg_{i:02d}_{gene}_vs_params.pdf')
            scatter_multi_plot(gene_log_data, gene_scatter_path)

# save top genes
top_pos_genes.to_csv(os.path.join(output_dir, 'top_pos_genes.csv'))
top_neg_genes.to_csv(os.path.join(output_dir, 'top_neg_genes.csv'))

# %%
# make heatmap of PCA loadings
plt.figure(figsize=(8, 8))
plt.imshow(V[:, :5], cmap='seismic', aspect='auto')
plt.colorbar()
plt.savefig(os.path.join(output_dir, 'pca_transform_matrix.png'))
plt.close()

# %%
# get Pearson correlations for top genes in top components vs param means
top_pos_gene_list = np.unique(top_pos_genes.loc[:, :num_top_comps - 1])
top_neg_gene_list = np.unique(top_neg_genes.loc[:, :num_top_comps - 1])
top_pc_gene_list = np.unique(
    np.concatenate((top_pos_gene_list, top_neg_gene_list), axis=None))

gene_vs_param_pearson_corrs = pd.DataFrame(
    0, columns=top_pc_gene_list, index=param_names)
gene_vs_param_pearson_corr_p_vals = pd.DataFrame(
    0, columns=top_pc_gene_list, index=param_names)

for param, gene in itertools.product(param_names, top_pc_gene_list):
    gene_idx = np.where(gene == gene_symbols)[0][0]
    corr, p_val = scipy.stats.pearsonr(
        log_data[gene_idx, sampled_cell_ids], sample_means[param])
    gene_vs_param_pearson_corrs.loc[param, gene] = corr
    gene_vs_param_pearson_corr_p_vals.loc[param, gene] = p_val

# save Pearson correlations to file
gene_vs_param_pearson_corrs.to_csv(
    os.path.join(output_dir, 'genes_vs_params', 'pearson_corrs.csv'))
gene_vs_param_pearson_corr_p_vals.to_csv(
    os.path.join(output_dir, 'genes_vs_params', 'pearson_corrs_p_vals.csv'))

# %%
# run regression analysis for genes vs sampled means of params
def run_regression_genes_vs_params(regressor_name, regression_genes, degree=1):
    '''Fit a regression model for gene expression vs sampled means of prams,
    with features raised to a specified degree'''
    print(f'Running {regressor_name} with degree {degree}')
    output_sub_dir = 'genes_vs_params_regression'

    r2_scores = pd.DataFrame(0, index=param_names, columns=regression_genes)
    mean_sq_errors = pd.DataFrame(0, index=param_names,
                                  columns=regression_genes)
    for gene in regression_genes:
        param_regressors = {}
        # get expression data, of shape num_cells * 1
        gene_idx = np.where(gene_symbols == gene)[0][0]
        X_gene = log_data[gene_idx, sampled_cell_ids, np.newaxis]
        # generate features from expression data
        if degree > 1:
            poly = PolynomialFeatures(degree)
            X = poly.fit_transform(X_gene)
        else:
            X = X_gene

        # perform regression for each param
        for param in param_names:
            regressor = regressor_classes[regressor_name]()
            y = sample_means[param]

            regressor.fit(X, y)
            y_pred = regressor.predict(X)
            param_regressors[param] = regressor

            # compute metrics
            r2_scores.loc[param, gene] = regressor.score(X, y)
            mean_sq_errors.loc[param, gene] = mean_squared_error(y, y_pred)

        # plot regression lines (curves)
        regression_scatter_path = os.path.join(
            output_dir, output_sub_dir,
            f'{r}_degree_{degree}_{gene}_scatter.pdf')
        scatter_multi_plot(X_gene, regression_scatter_path,
                           param_regressors=param_regressors, X_poly=X)

    # save metrics
    r2_scores_path = os.path.join(
        output_dir, output_sub_dir, f'{r}_degree_{degree}_scores.csv')
    r2_scores.to_csv(r2_scores_path, float_format='%.8f')

    mean_sq_errors_path = os.path.join(
        output_dir, output_sub_dir, f'{r}_degree_{degree}_mse.csv')
    mean_sq_errors.to_csv(mean_sq_errors_path, float_format='%.8f')

regressor_classes = {'linear': LinearRegression, 'huber': HuberRegressor}

for r, d in itertools.product(regressor_classes, range(1, 5)):
    run_regression_genes_vs_params(r, top_pc_gene_list, degree=d)

# %%
