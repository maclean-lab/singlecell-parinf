# Single-cell parameter inference for Ca<sup>2+</sup> pathway model

This repository contains code for Bayesian paramater inference for
Ca<sup>2+</sup> pathway model using No-U-Turn Sampler (NUTS) and post-sampling
analyses.

## Package requirement
The project requires Python 3.6 or newer. The following packages are also
required:
- PyStan 2.19
- NumPy
- SciPy
- statsmodels
- pandas
- scikit-learn
- ArviZ
- Matplotlib 3.4 or newer
- seaborn
- Jupyter notebook

## Project content
### Code demos
- `stan_calcium_mdoel_analysis.ipynb`: post-sampling analysis for a single
cell.
- `cell_chain_analysis.ipynb`: analysis for a cell chain.

### Modules
- `stan_helpers.py`: classes for sampling parameters for single cells and
analyzing sampled cells, as well as helper functions.
- `calcium_models.py`: ordinary differerential equations (ODEs) of each
Ca<sup>2+</sup> model variant, which are used when simulating Ca<sup>2+</sup>
trajectories during post-sampling analysis, as well as parameter names in the
models.

### Executable scripts
All Python scripts for generating data and results presented in our manuscript
in preparation are in the root folder.

If a script has a `main()` function, it is to be executed in command line. See
help using `python [script_name].py -h`.

If a script has no `main()` function, it is divided into code cells by `# %%`.
In Visual Studio Code or PyCharm (or any editor/IDE that supports this
feature), each code cell can be executed separately.

### Data
- `stan_models`: folder for Stan model specifications. Each subfolder contains
a model. Some models have associated prior specifications. If a model is
compiled by `compile_stan_model.py`, the compiled model will also be saved in
the same subfolder.
- `cell_lists`: folder for cell lists generated according to different
criteria.
- `cell_chain_example`: folder used by [code demos](#code-demos). The `samples`
subfolder stores posterior distribution and related NUTS information of 100
cells. The `cell_chain_analysis.ipynb` notebook will also create a subfolder
called `multi-sample-analysis` for its results.
- `stan_run_meta.json`: metadata of sampled cell chains.
- `stan_run_comparison_meta.json`: metadata for comparing sampled cell chains.
- `soptsc.mat`: saved variables from SoptSC, which includes a cell-cell
similarity matrix.

### Miscellaneous
- `deprecated`: archive folder for outdated scripts.
