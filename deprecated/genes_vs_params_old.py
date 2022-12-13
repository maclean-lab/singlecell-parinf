# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import os.path
import itertools
import math
import numpy as np
import scipy.stats
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import scanpy as sc
from anndata import AnnData

from stan_helpers import StanSessionAnalyzer

# %%
# set up
# root_cell_id, first_cell, last_cell = 5106, 5121, 4267
root_cell_id, first_cell, last_cell = 5085, 4982, 4553
stan_run_suffix = '-1'
use_highly_variable_genes = False
save_results = True

# %%
# load sampled parameters
# get cell list
cell_list_path = f'cell_lists/dfs_feature_100_root_{root_cell_id}_0.000_1.8.txt'
cell_list = pd.read_csv(cell_list_path, sep='\t')
first_cell_order = np.where(cell_list['Cell'] == first_cell)[0][0]
last_cell_order = np.where(cell_list['Cell'] == last_cell)[0][0]

param_names = ['sigma', 'L', 'Katp', 'KoffPLC', 'Vplc', 'Kip3', 'KoffIP3',
               'a', 'dinh', 'Ke', 'Be', 'd1', 'd5', 'epr', 'eta1', 'eta2',
               'eta3', 'c0', 'k3']
num_params = len(param_names)

# prepare output directories
stan_output_root = os.path.join(
    '../../result/',
    f'stan-calcium-model-100-root-{root_cell_id}{stan_run_suffix}')
if use_highly_variable_genes:
    output_dir = os.path.join(stan_output_root, 'hv-genes-vs-params')
else:
    output_dir = os.path.join(stan_output_root, 'genes-vs-params')
# create dirs if they don't exist
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
sub_output_dirs = ['genes-vs-params', 'genes-vs-params-regression',
                   'pca-top-genes', 'pcs-vs-params', 'pcs-vs-params-regression']
for d in sub_output_dirs:
    sub_output_full_path = os.path.join(output_dir, d)
    if not os.path.exists(sub_output_full_path):
        os.mkdir(sub_output_full_path)

# %%
# load gene expression
raw_data = pd.read_csv('vol_adjusted_genes_transpose.txt', sep='\t')

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
# load sampled parameters for each cell
sampled_cell_ids = []
sample_means = pd.DataFrame(columns=param_names)
for cell_order in trange(first_cell_order, last_cell_order + 1):
    # load samples
    cell_id = cell_list.loc[cell_order, 'Cell']
    cell_path = os.path.join(stan_output_root, f'cell-{cell_id:04d}')
    analyzer = StanSessionAnalyzer(cell_path, sample_source='arviz_inf_data',
                                   param_names=param_names)

    # compute sample means for cells with mixed chains (R_hat < 4.0)
    cell_sample_means = analyzer.get_sample_means(rhat_upper_bound=4.0)
    if cell_sample_means is not None:
        sampled_cell_ids.append(cell_id)
        sample_means = sample_means.append(cell_sample_means, ignore_index=True)

sampled_cell_ids = np.array(sampled_cell_ids)

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
                if isinstance(c, dict) or isinstance(c, pd.DataFrame):
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
# remove cells with large z scores
sample_mean_z_scores = sample_means.apply(scipy.stats.zscore, ddof=1)

# plot z-scores of sample means
if save_results:
    for gene_idx, gene in enumerate(tqdm(gene_symbols)):
        gene_scatter_path = os.path.join(output_dir, 'genes-vs-params',
                                        f'{gene}_z_scores.pdf')
        scatter_multi_plot(log_data[gene_idx, sampled_cell_ids],
                           gene_scatter_path, c=sample_mean_z_scores)

sample_mean_z_score_max = 3.0
is_cell_outlier = \
    (sample_mean_z_scores.abs() > sample_mean_z_score_max).any(axis=1)
sample_means = sample_means.loc[~is_cell_outlier, :]
sampled_cell_ids = sampled_cell_ids[~is_cell_outlier]
num_sampled_cells = sampled_cell_ids.size
cell_order_list = list(range(num_sampled_cells)) # orders of cells in the chain

# %%
# plot sample mean of params vs expression of every gene
if save_results:
    for gene_idx, gene in enumerate(tqdm(gene_symbols)):
        gene_scatter_path = os.path.join(output_dir, 'genes-vs-params',
                                        f'{gene}.pdf')
        scatter_multi_plot(log_data[gene_idx, sampled_cell_ids],
                           gene_scatter_path, c=cell_order_list)

# %%
# run PCA on log expression
num_comps = 50
num_top_comps = 10
pca = PCA(n_components=num_comps)
log_data_reduced = pca.fit_transform(log_data.T)
log_data_reduced_abs = np.abs(log_data_reduced)

# %%
# plot variance ratio explained by each component
if save_results:
    plt.plot(pca.explained_variance_ratio_, '.')
    plt.savefig(os.path.join(output_dir, 'pca_var_ratio.png'))
    plt.close()

# %%
# get Pearson correlations of top principal components vs params
pc_vs_param_pearson_corrs = pd.DataFrame(
    0, columns=param_names, index=range(num_top_comps))
pc_vs_param_pearson_corr_p_vals = pd.DataFrame(
    0, columns=param_names, index=range(num_top_comps))

for comp, param in itertools.product(range(num_top_comps), param_names):
    corr, p_val = scipy.stats.pearsonr(
        log_data_reduced_abs[sampled_cell_ids, comp], sample_means[param])
    pc_vs_param_pearson_corrs.loc[comp, param] = corr
    pc_vs_param_pearson_corr_p_vals.loc[comp, param] = p_val

# save Pearson correlations to file
if save_results:
    pc_vs_param_pearson_corrs.to_csv(
        os.path.join(output_dir, 'pcs-vs-params', 'pearson_corrs.csv'))
    pc_vs_param_pearson_corr_p_vals.to_csv(
        os.path.join(output_dir, 'pcs-vs-params', 'pearson_corrs_p_vals.csv'))

# %%
# make scatter plots for components vs params
if save_results:
    for comp in trange(num_top_comps):
        comp_data = log_data_reduced_abs[sampled_cell_ids, comp]
        comp_param_plot_path = os.path.join(output_dir, 'pcs-vs-params',
                                            f'comp_{comp}_vs_params.pdf')
        scatter_multi_plot(comp_data, comp_param_plot_path, c=cell_order_list)

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
    output_sub_dir = 'pcs-vs-params-regression'

    r2_scores = pd.DataFrame(0, index=range(num_top_comps), columns=param_names)
    mean_sq_errors = pd.DataFrame(0, index=range(num_top_comps),
                                  columns=param_names)
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
            r2_scores.loc[comp, param] = regressor.score(X, y)
            mean_sq_errors.loc[comp, param] = mean_squared_error(y, y_pred)

        # plot regression lines (curves)
        if save_results:
            regression_scatter_path = os.path.join(
                output_dir, output_sub_dir,
                f'{r}_degree_{degree}_comp_{comp}.pdf')
            scatter_multi_plot(X_comp, regression_scatter_path,
                            param_regressors=param_regressors, X_poly=X)

    # save metrics
    if save_results:
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

    if save_results and comp < num_top_comps:
        # plot top positive genes vs sampled means
        for i, gene in enumerate(top_pos_genes_comp):
            gene_idx = ranked_gene_indices[-i]
            gene_log_data = log_data[gene_idx, sampled_cell_ids]
            gene_scatter_path = os.path.join(
                output_dir, 'pca-top-genes',
                f'comp_{comp}_top_pos_{i:02d}_{gene}_vs_params.pdf')
            scatter_multi_plot(gene_log_data, gene_scatter_path)

        # plot top negative genes vs sampled means
        for i, gene in enumerate(top_neg_genes_comp):
            gene_idx = ranked_gene_indices[i]
            gene_log_data = log_data[gene_idx, sampled_cell_ids]
            gene_scatter_path = os.path.join(
                output_dir, 'pca-top-genes',
                f'comp_{comp}_top_neg_{i:02d}_{gene}_vs_params.pdf')
            scatter_multi_plot(gene_log_data, gene_scatter_path)

# save top genes
if save_results:
    top_pos_genes.to_csv(os.path.join(output_dir, 'top_pos_genes.csv'))
    top_neg_genes.to_csv(os.path.join(output_dir, 'top_neg_genes.csv'))

# %%
# make heatmap of PCA loadings
if save_results:
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
    0, columns=param_names, index=top_pc_gene_list)
gene_vs_param_pearson_corr_p_vals = pd.DataFrame(
    0, columns=param_names, index=top_pc_gene_list)

for gene, param in itertools.product(top_pc_gene_list, param_names):
    gene_idx = np.where(gene == gene_symbols)[0][0]
    corr, p_val = scipy.stats.pearsonr(
        log_data[gene_idx, sampled_cell_ids], sample_means[param])
    gene_vs_param_pearson_corrs.loc[gene, param] = corr
    gene_vs_param_pearson_corr_p_vals.loc[gene, param] = p_val

# save Pearson correlations to file
if save_results:
    gene_vs_param_pearson_corrs.to_csv(
        os.path.join(output_dir, 'genes-vs-params', 'pearson_corrs.csv'))
    gene_vs_param_pearson_corr_p_vals.to_csv(
        os.path.join(output_dir, 'genes-vs-params', 'pearson_corrs_p_vals.csv'))

# %%
# plot Pearson correlations as histogram for each param
if save_results:
    for param in param_names:
        plt.hist(gene_vs_param_pearson_corrs[param], bins=15)
        plt.savefig(os.path.join(output_dir, 'genes-vs-params',
                                 f'gene_vs_{param}_corr_hist.png'))
        plt.close()

#%%
# sort gene vs param correlations
num_gene_param_pairs = len(top_pc_gene_list) * num_params
sorted_gene_vs_param_pairs = pd.DataFrame(
    columns=['Gene', 'Parameter', 'Correlation', 'p-value'],
    index=range(num_gene_param_pairs))

for i, (gene, param) in enumerate(
        itertools.product(top_pc_gene_list, param_names)):
    corr = gene_vs_param_pearson_corrs.loc[gene, param]
    p_val = gene_vs_param_pearson_corr_p_vals.loc[gene, param]
    sorted_gene_vs_param_pairs.iloc[i, :] = [gene, param, corr, p_val]

sorted_gene_vs_param_pairs.sort_values(
    'Correlation', ascending=False, inplace=True, ignore_index=True,
    key=lambda x: np.abs(x))

if save_results:
    sorted_gene_vs_param_pairs.to_csv(
        os.path.join(output_dir, 'genes-vs-params', 'pearson_corrs_sorted.csv'))

# %%
# define pairs of genes and parameters with high correlations
high_corr_pairs = [('CAPN1', 'd5'), ('PPP1CC', 'L'), ('ITPRIPL2', 'k3'),
                   ('MSH2', 'd5'), ('PRKCI', 'd5'), ('PPP2CA', 'd5'),
                   ('PPP2CA', 'k3'), ('PRKCI', 'a'), ('PPP1CC', 'eta1'),
                   ('PPP1CC', 'd5'), ('ATP2B1', 'd5'), ('CCDC47', 'd5'),
                   ('RCN1', 'd5'), ('PPP1CC', 'a'), ('PPP2CB', 'd5'),
                   ('PPP3CA', 'd5'), ('PPP1CC', 'k3'), ('ATP2C1', 'd5')]
high_corr_pairs.sort()

# %%
# run regression analysis for genes vs sampled means of params
def run_regression_genes_vs_params(regressor_name, regression_genes, degree=1):
    '''Fit a regression model for gene expression vs sampled means of prams,
    with features raised to a specified degree'''
    print(f'Running {regressor_name} with degree {degree}')
    output_sub_dir = 'genes-vs-params-regression'

    r2_scores = pd.DataFrame(0, index=regression_genes, columns=param_names)
    mean_sq_errors = pd.DataFrame(0, index=regression_genes,
                                  columns=param_names)
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
            r2_scores.loc[gene, param] = regressor.score(X, y)
            mean_sq_errors.loc[gene, param] = mean_squared_error(y, y_pred)

            # save regressor
            gene_param_pair = (gene, param)
            if gene_param_pair in high_corr_pairs:
                regressor_trained[
                    (regressor_name, degree, gene, param)] = regressor

        # plot regression lines (curves)
        if save_results:
            regression_scatter_path = os.path.join(
                output_dir, output_sub_dir, f'{r}_degree_{degree}_{gene}.pdf')
            scatter_multi_plot(X_gene, regression_scatter_path,
                            param_regressors=param_regressors, X_poly=X)

    # save metrics
    if save_results:
        r2_scores_path = os.path.join(
            output_dir, output_sub_dir, f'{r}_degree_{degree}_scores.csv')
        r2_scores.to_csv(r2_scores_path, float_format='%.8f')

        mean_sq_errors_path = os.path.join(
            output_dir, output_sub_dir, f'{r}_degree_{degree}_mse.csv')
        mean_sq_errors.to_csv(mean_sq_errors_path, float_format='%.8f')

regressor_classes = {'linear': LinearRegression, 'huber': HuberRegressor,
                     'gamma': GammaRegressor}

regressor_trained = {}
for r, d in itertools.product(regressor_classes, range(1, 5)):
    run_regression_genes_vs_params(r, top_pc_gene_list, degree=d)

# %%
def plot_high_corr_pairs(output_path, regressor_name, num_rows=4, num_cols=2,
                         degree=1):
    """Make multiple scatter plots of some data vs param means in a PDF"""
    num_subplots_per_page = num_rows * num_cols
    num_plots = len(high_corr_pairs)
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

                # get data
                gene, param = high_corr_pairs[plot_idx]
                gene_idx = np.where(gene == gene_symbols)[0][0]
                X_data = log_data[gene_idx, sampled_cell_ids, np.newaxis]

                # make scatter plot
                plt.scatter(X_data, sample_means[param])
                corr = gene_vs_param_pearson_corrs.loc[gene, param]
                plt.title(f'{gene} vs {param}: {corr:.6f}')

                # plot regression line/curve
                if degree == 1:
                    X = X_data
                else:
                    poly = PolynomialFeatures(degree)
                    X = poly.fit_transform(X_data)
                regressor = regressor_trained[
                    (regressor_name, degree, gene, param)]
                sample_mean_pred = regressor.predict(X)
                plt.scatter(X_data, sample_mean_pred)

            plt.tight_layout()
            pdf.savefig()
            plt.close()

high_corr_pairs_scatter_path = os.path.join(
    output_dir, 'high_corr_pairs_scatter.pdf')
plot_high_corr_pairs(high_corr_pairs_scatter_path, 'huber')

# %%
num_param_param_pairs = num_params * (num_params - 1) // 2
sorted_param_param_corrs = pd.DataFrame(
    columns=['Param1', 'Param2', 'Correlation', 'p-value'],
    index=range(num_param_param_pairs))
for i, (p1, p2) in enumerate(itertools.combinations(param_names, 2)):
    corr, p_value = scipy.stats.pearsonr(sample_means[p1], sample_means[p2])
    sorted_param_param_corrs.iloc[i, :] = [p1, p2, corr, p_value]

sorted_param_param_corrs.sort_values(
    'Correlation', ascending=False, inplace=True, ignore_index=True,
    key=lambda x: np.abs(x))

# %%
