

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
        
        'VAE_config' : { "enc_dims": [256, 64], "dropout": 0.5, "anneal_cap": 0.2, "total_anneal_steps": 200000 },


        'Predefined_hyperparameters' : {
                            "MLP": {
                                "ML1M": {
                                    'learning_rate': 0.001, 'lambda_neg': 0.1414854294885049 , 'lambda_pos': 11.59908096547193,
                                    'alpha': 1, 'batch_size': 32, 'explainer_hidden_size': 64, 'epochs': 50 
                                },
                                "Yahoo": {
                                    'learning_rate': 0.01, 'lambda_neg': 0.19367009952856118, 'lambda_pos':12.40692505393434 ,
                                    'alpha': 1, 'batch_size': 64, 'explainer_hidden_size': 128, 'epochs': 50
                                },

                                "Pinterest": {
                                            'learning_rate': 0.001, 'lambda_neg': 0.705778173474644, 'lambda_pos':10.059416809308486 ,
                                            'alpha': 1, 'batch_size': 16, 'explainer_hidden_size': 16, 'epochs': 50},

                            },
                            
                            "VAE": {
                                "ML1M": {
                                    'learning_rate': 0.01, 'lambda_neg': 0.2535186589375764, 'lambda_pos': 2.456195926869126,
                                    'alpha': 1, 'batch_size': 64, 'explainer_hidden_size': 128, 'epochs': 50
                                },
                                "Yahoo": {
                                    'learning_rate': 0.01, 'lambda_neg':  1.9068379104210809, 'lambda_pos': 8.34930976885348,
                                    'alpha': 1, 'batch_size': 64, 'explainer_hidden_size': 32, 'epochs': 80
                                },
                                "Pinterest": {
                                    'learning_rate': 0.04, 'lambda_neg':  1.472868807603448 , 'lambda_pos': 6.3443735346179855,
                                    'alpha': 1, 'batch_size': 256, 'explainer_hidden_size': 32, 'epochs': 50
                                },
                            }
                        }



    
    }
