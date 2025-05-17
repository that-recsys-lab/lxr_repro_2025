
import argparse
from scripts.latent_factors import LatentFactorExplainer
from scripts.brute_force import BruteForceEvaluator
from scripts.popularity_explainer import PopularityExplainer
from scripts.LXR_10 import LXR_10
from scripts.LXR import LXR

import sys
import os

def run_explainer(args):
    dispatch = {
        "LF": lambda: LatentFactorExplainer(args.recommender, args.data, args.num_users, args.task),
        "POP": lambda: PopularityExplainer(args.recommender, args.data, args.num_users, args.task),
        "BF": lambda: BruteForceEvaluator(args.recommender, args.data, args.num_users, args.task),
        "LXR-10": lambda: LXR_10(args.recommender, args.data, args.num_users, args.task),
        "LXR": lambda: LXR(args.recommender, args.data, args.num_users, args.task)

    }

    if args.explainer not in dispatch:
        raise ValueError(f"Unknown explainer_name: {args.explainer}")

    explainer = dispatch[args.explainer]()
    explainer.predict()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--explainer', type=str, default="LXR-10", help='LF, POP, BF, LXR-10, LXR')
    parser.add_argument('--recommender', type=str, default="MLP", help='MLP or VAE')
    parser.add_argument('--data', type=str, default="ML1M", help='ML1M, Yahoo, Pinterest')
    parser.add_argument('--num_users', type=int, default=1000, help='Number of users to evaluate')
    parser.add_argument('--task', type=str, default="Top10", help='Top1 or Top10')
    args = parser.parse_args()
    run_explainer(args)
