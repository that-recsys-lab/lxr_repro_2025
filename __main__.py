
from scripts.latent_factors import LatentFactorExplainer
#from scripts.brute_force import BruteForceEvaluator
#from scripts.popularity_explainer import PopularityExplainer
#from scripts.LXR_10 import LXR_10
import sys
import os

explainer_name = "LF"
recommender_name = 'MLP'
data_name = 'ML1M'

def run_explainer():
    dispatch = {
        "LF": lambda: LatentFactorExplainer(recommender_name, data_name),
        "POP": lambda: PopularityExplainer(recommender_name, data_name),
        "BF": lambda: PopularityExplainer(recommender_name, data_name),
        "LXR-10": lambda: LXR_10(recommender_name, data_name, numb_rand_users=100)
    }

    if explainer_name not in dispatch:
        raise ValueError(f"Unknown explainer_name: {explainer_name}")

    explainer = dispatch[explainer_name]()
    explainer.predict()

if __name__ == "__main__":
    run_explainer()
