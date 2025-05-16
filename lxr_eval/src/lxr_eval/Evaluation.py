import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from lxr_eval.src.lxr_eval.help_functions import Help_Functions




class Evaluation(nn.Module):

    def __init__(self,data_name,recommender_name, explainer, recommender, kw_dict, k):
        super(Evaluation, self).__init__()
        self.explainer = explainer
        self.recommender = recommender
        self.kw_dict = kw_dict
        self.k = k
        self.hf=Help_Functions(recommender, data_name, recommender_name, kw_dict)

    def forward(self, user_tensor, i1, i1_index, i1_tensor):

        ''' i1 is the target item id and i1-indx is the index (rank) of target item '''

        user_hist_size = int(torch.sum(user_tensor))
        m1 = self.find_LXR_mask(user_tensor, i1_tensor)
        return self.process_sim_items(
            m1, i1, i1_index, user_tensor, user_hist_size)
    

    def find_LXR_mask(self, user_tensor, i1_tensor):
        m1 = self.explainer(user_tensor, i1_tensor)
        x_m1 = user_tensor * m1
        return {i: x_m1[i].item() for i in range(len(x_m1))}


    def mask_items(self, user_tensor, sim_items, total_items):
        mask = torch.zeros_like(user_tensor, dtype=torch.float32, device=user_tensor.device)
        indices = [item[0] for item in sim_items[:total_items]]
        mask[indices] = 1
        return user_tensor - mask


    def process_sim_items(self, m1, i1, i1_index, user_tensor, user_hist_size):
        sorted_m1 = list(sorted(m1.items(), key=lambda item: item[1], reverse=True))[:user_hist_size]
        total_items = 0
        for _ in sorted_m1:
            total_items += 1
            p = self.mask_items(user_tensor, sorted_m1, total_items)
            i1_rank =  self.hf.get_index_in_the_list(p, user_tensor, i1) + 1
            if i1_rank > i1_index + self.k:
                return total_items, sorted_m1
        return None, None