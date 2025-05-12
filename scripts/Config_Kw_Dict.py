

import os
from pathlib import Path
import torch



def get_kw_dict():
    export_dir = Path(os.getcwd())
    checkpoints_path = Path(export_dir, "checkpoints")

    return {
        'device': torch.device("mps" if torch.backends.mps.is_available() else 'cpu'),

        'output_type': {
            "VAE": "multiple",
            "MLP": "single"
        },

        'num_users': {
            "ML1M": 6037,
            "Yahoo": 13797,
            "Pinterest": 19155
        },

        'num_items': {
            "ML1M": 3381,
            "Yahoo": 4604,
            "Pinterest": 9362
        },

        'checkpoints_path': checkpoints_path,

        'recommender_path': {
            ("ML1M", "VAE"): checkpoints_path / "VAE_ML1M_0.0003_64.pt",
            ("ML1M", "MLP"): checkpoints_path / "MLP1_ML1M_0.0076_256_7.pt",

            ("Yahoo", "VAE"): checkpoints_path / "VAE_Yahoo_128.pt",
            ("Yahoo", "MLP"): checkpoints_path / "MLP2_Yahoo_0.0083_128_1.pt",

            ("Pinterest", "VAE"): checkpoints_path / "VAE_Pinterest_12_18_0.0001_256.pt",
            ("Pinterest", "MLP"): checkpoints_path / "MLP_Pinterest_0.0062_512_21_0.pt"
        },

        'hidden_dim': {
            ("ML1M", "VAE"): None,
            ("ML1M", "MLP"): 32,

            ("Yahoo", "VAE"): None,
            ("Yahoo", "MLP"): 32,

            ("Pinterest", "VAE"): None,
            ("Pinterest", "MLP"): 512
        },
        
        'VAE_config' : { "enc_dims": [256, 64], "dropout": 0.5, "anneal_cap": 0.2, "total_anneal_steps": 200000 }



    
    }
