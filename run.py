import argparse
import os
from scripts.latent_factors import LatentFactorExplainer
from scripts.brute_force import BruteForceEvaluator
from scripts.popularity_explainer import PopularityExplainer
from scripts.LXR_10 import LXR_10
from scripts.LXR import LXR

def read_args():
    parser = argparse.ArgumentParser(
        description="LXR Reproducibility Runner for Explanation Experiments"
    )

    parser.add_argument("--explainer", type=str, help="LF, POP, BF, LXR, or LXR-10")
    parser.add_argument("--recommender", type=str, default="MLP", help="MLP or VAE")
    parser.add_argument("--data", type=str, default="ML1M", help="ML1M, Yahoo, or Pinterest")
    parser.add_argument("--num_users", type=int, default=1000, help="Number of users to evaluate")
    parser.add_argument("--task", type=str, default="Top10", help="Top1 or Top10 task")

    args = parser.parse_args()
    return vars(args)

def dispatch_explainer(explainer_name, recommender, data, num_users, task):
    dispatch = {
        "LF": LatentFactorExplainer,
        "POP": PopularityExplainer,
        "BF": BruteForceEvaluator,
        "LXR-10": LXR_10,
        "LXR": LXR
    }

    if explainer_name not in dispatch:
        raise ValueError(f"Unknown explainer: {explainer_name}")

    return dispatch[explainer_name](recommender, data, num_users, task)

def main():
    args = read_args()
    explainer = dispatch_explainer(
        args["explainer"], args["recommender"], args["data"], args["num_users"], args["task"]
    )
    explainer.predict()

if __name__ == "__main__":
    main()
