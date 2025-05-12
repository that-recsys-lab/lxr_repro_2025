import numpy as np
import os
import pandas as pd
import pickle
import torch
from itertools import combinations

from help_functions import get_index_in_the_list #not sure where help_functions is coming from -- is this supposed to be an internal module?


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
export_dir = os.getcwd()

THRESHOLD_VAL = 5 #verify good constant name; PROVIDE DESCRIPTION
GLOBAL_INDEX = 49 #verify good constant name; PROVIDE DESCRIPTION
DEFAULT_K = 10 #how many places away target item must be; CHECK is this k in latent factors and popularity explainer?

'''
    Evaluates brute force perturbations against LXR's perturbations.
'''
class BruteForceEvaluator:
    def __init__(self, data_name, recommender_name, recommender, kw_dict):
        self.data_name = data_name
        self.recommender_name = recommender_name
        self.recommender = recommender
        self.kw_dict = kw_dict
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self._load_lxr_outputs()

    '''
        Loads LXR outputs from .pkl files.
    '''
    def _load_lxr_outputs(self):
        d = self.data_name
        r = self.recommender_name
        self.MPRR_P = self._load_pickle(f'MPRR_Percent_LXR_{d}_{r}.pkl')
        self.MPRR_R = self._load_pickle(f'MPRR_Raw_LXR_{d}_{r}.pkl')
        self.Mask = self._load_pickle(f'Mask_LXR_{d}_{r}.pkl')
        self.User_ID = self._load_pickle(f'User_ID_LXR_{d}_{r}.pkl')
        self.User_Tensor = self._load_pickle(f'User_Tensor_LXR_{d}_{r}.pkl')
        self.Targ_test = self._load_pickle(f'Targ_Test_LXR_{d}_{r}.pkl')
        self.Targ_INDX = self._load_pickle(f'Targ_INDX_LXR_{d}_{r}.pkl')

    '''
        Loads pickle file.
        filename: name of pickle file to be loaded
        returns: object of loaded file
    '''
    def _load_pickle(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    '''
        Creates subsets of LXR perturbation set and predicts if ranking changes with smaller set.
        Prints out MPNR achieved for LXR and brute force search.
    '''
    def evaluate(self):
        indices = [i for i, val in enumerate(self.MPRR_R[GLOBAL_INDEX]) if val < THRESHOLD_VAL]
        MPNR_lxr = []
        MPNR_bf = []

        for i in indices:
            user_tensor = self.User_Tensor[GLOBAL_INDEX][i].to(self.device)
            pert = self.Mask[GLOBAL_INDEX][i].to(self.device)
            vec = torch.nonzero(pert).view(-1).tolist()
            all_combinations = [list(c) for r in range(1, len(vec) + 1) for c in combinations(vec, r)]
            MPNR_lxr.append(self.MPRR_R[GLOBAL_INDEX][i])

            for j in all_combinations:
                mask = torch.zeros_like(user_tensor, device=self.device)
                mask[j] = 1
                p = user_tensor - mask # choose more informative name 
                indx = get_index_in_the_list(p, user_tensor, self.Targ_test[GLOBAL_INDEX][i], self.recommender, **self.kw_dict) + 1

                if indx > DEFAULT_K + self.Targ_INDX[GLOBAL_INDEX][i]:
                    MPNR_lxr.append(self.MPRR_R[GLOBAL_INDEX][i])
                    MPNR_bf.append(len(j))
                    break

        mean_lxr = np.mean(MPNR_lxr) if MPNR_lxr else 0.0
        mean_bf = np.mean(MPNR_bf) if MPNR_bf else 0.0

        print(f'MPNR achieved by the LXR model ({self.recommender_name} and {self.data_name}):', mean_lxr, flush=True)
        print(f'MPNR achieved by brute force search ({self.recommender_name} and {self.data_name}):', mean_bf, flush=True)

        return mean_lxr, mean_bf








