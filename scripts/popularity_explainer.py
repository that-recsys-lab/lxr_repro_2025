import numpy as np
import os
import torch

from help_functions import get_index_in_the_list

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
export_dir = os.getcwd()

SEED = 42 #random seed for random and torch 
DEFAULT_K = 10 #how many places away target item must be; CHECK


'''
    Makes explanations based on popularity of items.
'''
class PopularityExplainer:
    def __init__(self, data_name, recommender_name, recommender, kw_dict):
        self.data_name = data_name
        self.recommender_name = recommender_name
        self.recommender = recommender
        self.kw_dict = kw_dict
        self.pop_dict = kw_dict['pop_dict']
        self.device = kw_dict['device']

    '''
        ADD WHAT FUNCTION DOES HERE (before printing)
        Prints out MPNR and coverage achieved for the popularity explainer.
    '''
    def evaluate(self, test_array, targ_test):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        num_of_rand_users = test_array.shape[0] 

        random_rows = np.random.choice(test_array.shape[0], num_of_rand_users, replace=False)
        random_sampled_array = test_array[random_rows]
        total_pert_sizes = []
        for i in range(num_of_rand_users):
            user_id = random_sampled_array[i][-1]
            user_tensor = torch.Tensor(random_sampled_array[i][:-1]).to(self.device)
            user_hist_size = int(torch.sum(user_tensor))
            targ_item = np.random.choice(targ_test[user_id])
            targ_idx = list(targ_test[user_id]).index(targ_item)

            total_items = self._calculate_explanation(user_tensor, targ_item, targ_idx, user_hist_size)
            if total_items is not None:
                total_pert_sizes.append(total_items)

        print(f'MPNR for Popularity Explainer on {self.data_name} and {self.recommender_name} is ', np.mean(total_pert_sizes))
        print(f'Coverage for Popularity Explainer on {self.data_name} and {self.recommender_name} is ', (len(total_pert_sizes) * 100) / num_of_rand_users)

    '''
        Calculates explanation (how many items should be removed to achieve target).
        user_tensor: tensor representing the user
        targ_id: id of the target item
        targ_idx: index of the target item
        user_hist_size: how many items to consider from the user profile
        returns: number of items to be removed
    '''
    def _calculate_explanation(self, user_tensor, targ_id, targ_idx, user_hist_size):
        pop_sim_items = self._find_POP_mask(user_tensor)
        total_items = self._process_sim_items(user_tensor, targ_id, targ_idx, user_hist_size, pop_sim_items)
        return total_items

    '''
        Computes popularity similarity (mask) between user and items
        user_tensor: tensor representing the user
        returns: user/item similarity dictionary
    '''
    def _find_POP_mask(self, user_tensor):
        scores = self._explainer(user_tensor)
        pop_x_masked = user_tensor * scores
        pop_item_sim_dict = {i: pop_x_masked[i].item() for i in range(len(pop_x_masked))}    
        return pop_item_sim_dict

    '''
        MORE DESCRIPTION HERE OF WHAT FUNCTION IS DOING 
        user_tensor: tensor representing the user
        targ_id: id of the target item
        targ_idx: index of the target item
        user_hist_size: how many items to consider from the user profile
        sim_items: user/item similarity dictionary
        returns: number of items to be removed
    '''
    def _process_sim_items(self, user_tensor, targ_id, targ_idx, user_hist_size, sim_items):
        sorted_sim_items = list(sorted(sim_items.items(), key=lambda item: item[1], reverse=True))[:user_hist_size]

        total_items = 0
        for i in sorted_sim_items: #same comment as in latent_factors about use of i
            total_items += 1
            POS_masked = self._mask_items(user_tensor, sorted_sim_items, total_items)
            kw_dict = self.kw_dict #suggest directly using the target index instead of passing in the whole dictionary
            targ_rank = get_index_in_the_list(POS_masked, user_tensor, targ_id, self.recommender, **kw_dict) + 1
        
            if (targ_rank > DEFAULT_K + targ_idx):
                return total_items
    
        return None
        
    '''
        MORE DESCRIPTION HERE OF WHAT FUNCTION IS DOING
        user_tensor: tensor representing the user
        returns: ???
    '''
    def _explainer(self, user_tensor):
        score_tensor = torch.zeros_like(user_tensor)
        indx = torch.nonzero(user_tensor.squeeze())
        for i in indx:
            idx = i.item()
            score_tensor.squeeze()[idx] = self.pop_dict[idx].item()
        return score_tensor
    
    '''
        MORE DESCRIPTION HERE OF WHAT FUNCTION IS DOING (one hot encoding? other?)
        user_tensor: tensor representing the user
        sim_items: user/item similarity dictionary
        total_items:
        returns: mask of user_tensor
    '''
    def _mask_items(self, user_tensor, sim_items, total_items):
        mask = torch.zeros_like(user_tensor, dtype = torch.float32, device = self.device)
        indices = [item[0] for item in sim_items[:total_items]]
        mask[indices] = 1
        return user_tensor - mask
