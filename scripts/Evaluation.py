import torch
import torch.nn as nn

from help_functions import  get_index_in_the_list

'''
    CLASS DESCRIPTION
'''
class Evaluation(nn.Module):
    def __init__(self, explainer, recommender, kw_dict, k):
        super(Evaluation, self).__init__()
        self.explainer = explainer
        self.recommender = recommender
        self.kw_dict = kw_dict
        self.k = k

    '''
        DESCRIPTION OF METHOD
        user_tensor: tensor representing the user
        i1: target item id
        i1_index: index (rank) of target item
        i1_tensor: ???
        returns: ???
    '''
    def forward(self, user_tensor, i1, i1_index, i1_tensor):
        user_hist_size = int(torch.sum(user_tensor))
        m1 = self._find_LXR_mask(user_tensor, i1_tensor)
        return self._process_sim_items(m1, i1, i1_index, user_tensor, user_hist_size)
    
    '''
        DESCRIPTION OF METHOD
        user_tensor: tensor representing the user
        i1_tensor: 
        returns:
    '''
    def _find_LXR_mask(self, user_tensor, i1_tensor):
        m1 = self.explainer(user_tensor, i1_tensor)
        x_m1 = user_tensor * m1
        return {i: x_m1[i].item() for i in range(len(x_m1))}

    '''
        DESCRIPTION OF METHOD
        m1: ??
        i1: target item id
        i1_index: index (rank) of target item
        user_tensor: tensor representing the user
        user_hist_size: 
        returns:
    '''
    def _process_sim_items(self, m1, i1, i1_index, user_tensor, user_hist_size):
        sorted_m1 = list(sorted(m1.items(), key=lambda item: item[1], reverse=True))[:user_hist_size]
        total_items = 0
        for _ in sorted_m1:
            total_items += 1
            p = self._mask_items(user_tensor, sorted_m1, total_items)
            i1_rank = get_index_in_the_list(p, user_tensor, i1, self.recommender, **self.kw_dict) + 1
            if i1_rank > i1_index + self.k:
                return total_items
        return None
    
    '''
        DESCRIPTION OF METHOD
        user_tensor: tensor representing the user
        sim_items:
        total_items: 
        returns:
    '''
    def _mask_items(self, user_tensor, sim_items, total_items):
        mask = torch.zeros_like(user_tensor, dtype=torch.float32, device=user_tensor.device)
        indices = [item[0] for item in sim_items[:total_items]]
        mask[indices] = 1
        return user_tensor - mask