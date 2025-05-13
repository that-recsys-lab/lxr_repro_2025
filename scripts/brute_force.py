
from itertools import combinations
import numpy as np
import os
from pathlib import Path
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
export_dir = os.getcwd()
import pickle
import torch
import torch.nn as nn
from scripts.help_functions import Help_Functions
from scripts.Config_Kw_Dict import get_kw_dict
from scripts.recommender.recommenders_architecture import MLP, VAE





class BruteForceEvaluator:
    def __init__(self, recommender_name, data_name):

        self.kw_dict = get_kw_dict()
        self.data_name = data_name
        self.recommender_name = recommender_name
        self.recommender = self.load_recommender(recommender_name)
        self._load_lxr_outputs()
        self.hf=Help_Functions(self.recommender,data_name, recommender_name, self.kw_dict )

    def _load_pickle(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

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


    def load_recommender(self,recommender_name):
        kw_dict= self.kw_dict
        #data_name=self.data_name
        if recommender_name=='MLP':

            recommender = MLP(self.data_name, **kw_dict)
        elif recommender_name=='VAE':
            recommender = VAE(self.data_name, **kw_dict)
        recommender_checkpoint = torch.load(Path(kw_dict['checkpoints_path'], kw_dict['recommender_path'][(self.data_name, recommender_name)] ), map_location=kw_dict['device'])
        recommender.load_state_dict(recommender_checkpoint)
        recommender.eval()
        for param in recommender.parameters():
            param.requires_grad= False
        return recommender

        
    def evaluate(self):
        indices = [i for i, val in enumerate(self.MPRR_R[49]) if val < 5]
        MPNR_lxr = []
        MPNR_bf = []
        print(f'======================== Brute Force explainer run for {self.data_name} and {self.recommender_name}========================')

        for j in indices:
            user_tensor = self.User_Tensor[49][j].to(self.device)
            pert = self.Mask[49][j].to(self.device)
            vec = torch.nonzero(pert).view(-1).tolist()
            all_combinations = [list(c) for r in range(1, len(vec) + 1) for c in combinations(vec, r)]
            MPNR_lxr.append(self.MPRR_R[49][j])

            for i in all_combinations:
                mask = torch.zeros_like(user_tensor, device=self.device)
                mask[i] = 1
                p = user_tensor - mask
                indx = self.hf.get_index_in_the_list(p, user_tensor, self.Targ_test[49][j], self.recommender, **self.kw_dict) + 1

                if indx > 10 + self.Targ_INDX[49][j]:
                    MPNR_lxr.append(self.MPRR_R[49][j])
                    MPNR_bf.append(len(i))
                    break

        mean_lxr = np.mean(MPNR_lxr) if MPNR_lxr else 0.0
        mean_bf = np.mean(MPNR_bf) if MPNR_bf else 0.0

        print(f'MPNR achieved by the LXR model ({self.recommender_name} and {self.data_name}):', mean_lxr, flush=True)
        print(f'MPNR achieved by brute force search ({self.recommender_name} and {self.data_name}):', mean_bf, flush=True)

        return mean_lxr, mean_bf








