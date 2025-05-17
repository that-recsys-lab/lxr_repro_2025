# lxr_repro_2025

This repository contains the code and data necessary to reproduce the results from the paper "Counterfactual Profile Perturbations for Recommendation Explanations: A Reproducibility Study".

Paper submitted to the Reproducibility Track of the ACM RecSys conference 2025.

It is based on initial work that appears in 

Barkan, O., Bogina, V., Gurevitch, L., Asher, Y., & Koenigstein, N. Learning Counterfactual Explanations for Recommender Systems. In The Web Conference 2024.

https://dl.acm.org/doi/abs/10.1145/3589334.3645560

Many thanks to the authors of this paper for their assistance in our reproducibility work.


## Create a virtual environment and install dependencies

It is recommended to use a virtual environment to manage dependencies. "Requirements.txt" contains the packages we used for this project. Run the following commands:

```bash

# Create a virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install required packages
uv pip install -r requirements.txt
```