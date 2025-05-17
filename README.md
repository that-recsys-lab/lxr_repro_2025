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

## Project Structure
```
lxr_repro_2025/
├── scripts/                      # Scripts for Explainers
│   ├── __init__.py
│   ├── latent_factors.py         # Latent Factors Explainer
│   ├── brute_force.py            # Brute Force Search
│   ├── popularity_explainer.py   # Popularity Explainer
│   ├── LXR_10.py                 # Training LXR10 Explainer
│   ├── LXR.py                    # Training LXR Explainer
│   └── load_data.py              # Loading data for training the explainers
├── data/
│   ├── ML1M/                     # Folder for MovieLens dataset
│   ├── Yahoo/                    # Folder for Yahoo dataset
│   └── Pinterest/                # Folder for Pinterest dataset
├── lxr/
│   ├── Explainer.py              # LXR Explainer (Structure)
│   └── LXR_Loss.py               # LXR Loss Function
├── lxr_eval/
│   ├── Evaluation.py             # Evaluating LXR Explainer
│   └── help_functions.py         # Helper functions for generating recommendations
├── notebooks/                    # Jupyter notebooks for development and analysis
│   └── plot.ipynb                # For plotting line plots (fig2)
├── outputs/                      # Outputs like plots, evaluation metrics, tables
│   ├── figures/                  # For plots
│   ├── results/                  # For .pkl, .csv, .json result files
│   └── logs/                     # Optional logs from runs
├── run.py                       # For running experiments
├── requirements.txt              # Python dependencies
├── README.md                     # Project overview and instructions
```

## WANDP
This code uses the "Weights and Biases" library for tracking experiment parameters: For more information see [https://docs.wandb.ai/quickstart/](https://docs.wandb.ai/quickstart/). In order to train the explainer, you will need to create an account and input the API key when running the tuning section of the explainer notebooks.

## How to run the code?
Training the explainer can take several hours, depending on your server’s capabilities. If time is limited, consider evaluating a smaller sample of users. In our experiments, we used the full test set; however, evaluating only 500 users significantly reduced the runtime. Keep in mind that using fewer users may result in less accurate results than those reported in the paper.
To run the code, you need to specify the dataset, the recommender system, the explainer, the number of users, and the task. You can choose appropriate values for each of these parameters based on your requirements. For more details on configuring these parameters, please refer to the paper. To run the brute-force search, you must first need to train the LXR and LF explantions.
```
explainer ('LF', 'POP', 'BF', 'LXR-10', 'LXR')
recommender ('MLP', 'VAE')
data ("ML1M', 'Yahoo', 'Pinterest')
num_users (Valid integer value)
task ('Top1', 'Top10')
```
## Example
```
python run.py --explainer LXR-10 --recommender MLP --data ML1M --num_users 500 --task Top10
```
