
from scripts.latent_factors import LatentFactorExplainer
#from scripts.brute_force import BruteForceEvaluator
from scripts.popularity_explainer import PopularityExplainer
from scripts.LXR_10 import LXR_10
from scripts.LXR import LXR

import sys
import os

explainer_name = "LXR"      ## Can be  "LF", "POP", "BF", "LXR-10" "LXR"
recommender_name = 'MLP'    ## Can be "MLP" and "VAE"
data_name = 'ML1M'          ## Can be "ML1M", "Yahoo" and "Pinterest"
numb_rand_users=100         ## Number of users for  evaluation. We used all users for all experiments.



def run_explainer():
    dispatch = {
        "LF": lambda: LatentFactorExplainer(recommender_name, data_name),
        "POP": lambda: PopularityExplainer(recommender_name, data_name),
        "BF": lambda: PopularityExplainer(recommender_name, data_name),
        "LXR-10": lambda: LXR_10(recommender_name, data_name, numb_rand_users),
        "LXR": lambda: LXR(recommender_name, data_name, numb_rand_users)

    }

    if explainer_name not in dispatch:
        raise ValueError(f"Unknown explainer_name: {explainer_name}")

    explainer = dispatch[explainer_name]()
    explainer.predict()

if __name__ == "__main__":
    run_explainer()
