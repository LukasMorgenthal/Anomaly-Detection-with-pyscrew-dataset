# Modular benchmark scaffold

This folder contains the shared code that used to live inside the notebook.

Notebook should stay responsible for:

- loading the data
- light preprocessing and filtering
- choosing the target column
- calling `main.py` or `benchmark_modular.experiments.run_benchmark`
- displaying tables and plots

Reusable code moved here:

- model definitions
- training loops
- metrics
- benchmark runners
