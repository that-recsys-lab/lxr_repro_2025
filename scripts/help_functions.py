import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from load_data import load_data
from pathlib import Path
import os


class Help_Functions:
    def __init__(self, recommender, data_name, recommender_name, kw):


        self.recommender = recommender
        self.data_name=data_name
        self.recommender_name=recommender_name
        self.device = kw['device']
        self.num_items = kw['num_items'][data_name]
        dict_data=load_data(data_name, recommender_name,kw )
        self.all_items_tensor = dict_data['all_items_tensor']
        self.items_array = dict_data['items_array']
        self.pop_array = dict_data['pop_array']
        self.static_test_data = dict_data['static_test_data']
        self.output_type=kw['output_type'][recommender_name]
    
    
    
    
    
         
    
    
    
    def sample_indices(self, data):
        matrix = np.array(data)[:, :self.num_items] # keep only items columns, remove demographic features columns
        zero_indices, one_indices = [], []

        for i, row in enumerate(matrix):
            zero_idx = np.where(row == 0)[0]
            one_idx = np.where(row == 1)[0]

            probs = self.pop_array[zero_idx]
            probs = probs / np.sum(probs)

            sampled_zero = np.random.choice(zero_idx, p=probs)   # sample negative interactions according to items popularity
            sampled_one = np.random.choice(one_idx)    # sample positive interactions from user's history

            data.iloc[i, sampled_one] = 0
            zero_indices.append(sampled_zero)
            one_indices.append(sampled_one)

        data['pos'] = one_indices
        data['neg'] = zero_indices
        return np.array(data)

    def recommender_run(self, user_tensor, item_tensor=None, item_id=None, wanted_output='single'):
        if self.output_type == 'single':
            if wanted_output == 'single':
                return self.recommender(user_tensor, item_tensor)
            else:
                return self.recommender(user_tensor, item_tensor).squeeze()
        else:
            if wanted_output == 'single':
                return self.recommender(user_tensor).squeeze()[item_id]
            else:
                return self.recommender(user_tensor).squeeze()

    def get_top_k(self, user_tensor, original_user_tensor):
        item_prob_dict = {}
        output_model = self.recommender_run(user_tensor, self.all_items_tensor, None,wanted_output='vector').cpu().detach().numpy().tolist()
        original_user_vector = original_user_tensor.cpu().numpy()[:self.num_items]
        catalog = np.ones_like(original_user_vector) - original_user_vector
        output = catalog * output_model

        for i in range(len(output)):
            if catalog[i] > 0:
                item_prob_dict[i] = output[i]

        return dict(sorted(item_prob_dict.items(), key=lambda item: item[1], reverse=True))

    def get_index_in_the_list(self, user_tensor, original_user_tensor, item_id):
        top_k_list = list(self.get_top_k(user_tensor, original_user_tensor).keys())
        return top_k_list.index(item_id)





    def get_user_recommended_item(self, user_tensor):
        user_res = self.recommender_run(user_tensor, self.all_items_tensor,None, wanted_output='vector')[:self.num_items]
        user_tensor = user_tensor[:self.num_items]
        catalog = torch.ones_like(user_tensor) - user_tensor
        recommendations = torch.mul(user_res, catalog)
        sorted_recommendations = torch.argsort(recommendations, descending=True)
        return sorted_recommendations[:10]




    def get_ndcg(self, ranked_list, target_item):

        if target_item not in ranked_list:
            return 0.0
        target_idx = torch.tensor(ranked_list.index(target_item), device=device)
        dcg = torch.reciprocal(torch.log2(target_idx + 2))
        return dcg.item()

    def recommender_evaluations(self):
        n = len(self.static_test_data)
        counter_10 = counter_50 = counter_100 = RR = PR = 0

        for i in range(n):
            entry = self.static_test_data[i]
            item_id = entry[-2]
            item_tensor = self.items_array[item_id]
            user_tensor = torch.tensor(entry[:-2], device=self.device)
            user_tensor[item_id] = 0

            index = self.get_index_in_list(user_tensor, user_tensor, item_id) + 1
            if index <= 10: counter_10 += 1
            if index <= 50: counter_50 += 1
            if index <= 100: counter_100 += 1
            RR += 1 / index
            PR += index / self.num_items

        return (
            counter_10 / n,
            counter_50 / n,
            counter_100 / n,
            RR / n,
            100 * PR / n,
        )
