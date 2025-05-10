import numpy as np
import os
import torch

from help_functions import get_index_in_the_list

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
export_dir = os.getcwd()
SEED = 42


class PopularityExplainer:
    def __init__(self, data_name, recommender_name, recommender, kw_dict ):
            self.data_name=data_name
            self.recommender_name=recommender_name
            self.recommender=recommender
            self.kw_dict=kw_dict
            self.pop_dict=kw_dict['pop_dict']
            self.device=kw_dict['device']
            self.items_array=kw_dict['items_array']
    
    def evaluate(self, test_array, targ_test ):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        num_of_rand_users = test_array.shape[0] # number of users for evaluations

        random_rows = np.random.choice(test_array.shape[0], num_of_rand_users, replace=False)
        random_sampled_array = test_array[random_rows]
        total_pert=[] ## size of perturbations
        for j in range(num_of_rand_users):
            user_id = random_sampled_array[j][-1]
            user_tensor = torch.Tensor(random_sampled_array[j][:-1]).to(self.device)
            user_hist_size=int(torch.sum(user_tensor))
            targ_item = np.random.choice(targ_test[user_id])
            targ_idx=list(targ_test[user_id]).index(targ_item)

            targ_vector = self.items_array[targ_item]
            targ_tensor = torch.Tensor(targ_vector).to(self.device)
            total_items= self._calculate_explanation(user_tensor, targ_item, targ_idx, user_hist_size, k=10)
            if total_items is not None:
                total_pert.append(total_items)

        print(f'MPNR for Popularity Explainer on {self.data_name} and {self.recommender_name} is ', np.mean(total_pert))
        print(f'Coverage for Popularity Explainer on {self.data_name} and {self.recommender_name} is ', len(total_pert)*100/num_of_rand_users)

    def _calculate_explanation(self, user_tensor, targ_id, targ_idx, user_hist_size, k):
        pop_sim_items = self._find_POP_mask(user_tensor)
        total_items=self._process_sim_items(pop_sim_items, targ_id,targ_idx, user_tensor, user_hist_size,k)
        return total_items

    def _find_POP_mask(self, user_tensor):
        """" For finding masks based on popularity score """
        scores=self._explainer(user_tensor)
        pop_x_masked = user_tensor* scores
        pop_item_sim_dict = {i: pop_x_masked[i].item() for i in range(len(pop_x_masked))}    
        return pop_item_sim_dict

    def _process_sim_items(self, sim_items, targ_id,targ_idx, user_tensor, user_hist_size,k):
        sorted_sim_items = list(sorted(sim_items.items(), key=lambda item: item[1], reverse=True))[:user_hist_size]

        total_items = 0
        for i in sorted_sim_items:
            total_items += 1
            ## masking user profile 
            POS_masked = self._mask_items(user_tensor, sorted_sim_items, total_items)
            ##index of first item and second item after masking
            kw_dict=self.kw_dict
            targ_rank = get_index_in_the_list(POS_masked, user_tensor, targ_id, self.recommender, **kw_dict) + 1
        
            if (targ_rank > k +targ_idx):
                return total_items
    
        return None
        
    def _explainer(self, user_tensor):
        """ Explainer Scores"""
        score_tensor=torch.zeros_like(user_tensor)
        indx=torch.nonzero(user_tensor.squeeze())
        for i in indx:
            idx=i.item()
            score_tensor.squeeze()[idx]=self.pop_dict[idx].item()
        return score_tensor
    
    def _mask_items(self, user_tensor, sim_items, total_items):
        mask = torch.zeros_like(user_tensor, dtype=torch.float32, device=self.device)
        indices = [item[0] for item in sim_items[:total_items]]
        mask[indices] = 1
        return user_tensor - mask
