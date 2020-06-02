Experiments
===

`notebooks`: a directory with Jupyter notebooks each containing one experiment,
rerunning the experiments is done by running all cells of a notebook

`pmlib`: a Python package containing functions and classes that are used
withing the Jupyter notebooks, this package must be installed in Python
environment in order to rerun the experiments

`pyproject.toml`, `poetry.lock`: files for Poetry package manager to install
the same Python dependencies as used when running the experiments, see
[Poetry documentation](https://python-poetry.org/docs/) for a guide how to set
up the environment using Poetry
