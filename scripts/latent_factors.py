import numpy as np
import os
import torch

from help_functions import get_index_in_the_list

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
export_dir = os.getcwd()
SEED = 42

'''
    Makes explanations based on latent factors between items and users.
'''
class LatentFactorExplainer:
    def __init__(self,MF_recommender,recommender,kw_dict,targ_test, test_array, recommender_name, data_name):
        self.user_embeddings = MF_recommender.users_fc.weight.detach().cpu().numpy()
        self.item_embeddings = MF_recommender.items_fc.weight.detach().cpu().numpy()
        self.recommender=recommender
        self.kw_dict=kw_dict
        self.targ_test=targ_test
        self.test_array=test_array
        self.recommender_name=recommender_name
        self.data_name=data_name
        self.items_array=kw_dict['items_array']
        self.device=kw_dict['device']

    '''
        ADD WHAT FUNCTION DOES HERE (before printing)
        Prints out MPNR Row and coverage achieved for latent factor similarity.
    '''
    def evaluate(self):
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        MPRR_Row, MPRR_Percent=[],[]
        num_of_rand_users = self.test_array.shape[0]   # number of users for evaluations 
        random_rows = np.random.choice(self.test_array.shape[0], num_of_rand_users, replace=False)
        random_sampled_array = self.test_array[random_rows]

        for j in range(random_sampled_array.shape[0]):
            user_id = random_sampled_array[j][-1]
            user_tensor = torch.Tensor(random_sampled_array[j][:-1]).to(self.device)
            targ = np.random.choice(self.targ_test[user_id])
            targ_indx=list(self.targ_test[user_id]).index(targ)
            p = self._calculate_explanation(user_tensor, targ, targ_indx, k=10)

            if p is not None:
                MPRR_Row.append(p)
                MPRR_Percent.append(p/int(torch.sum(user_tensor)))

        print(f'MPNR Row for latent factors similarity for {self.data_name} and {self.recommender_name}:', np.mean(MPRR_Row))
        print(f'Coverage for latent factors similarity for {self.data_name} and {self.recommender_name}:', len(MPRR_Row)*100/num_of_rand_users)

    '''
        Calculates explanation (how many items should be removed to achieve target).
        user_tensor: tensor representing the user
        targ_id: id of the target item
        targ_idx: index of the target item
        k: how many places away target item must be
        returns: number of items to be removed
    '''
    def _calculate_explanation(self, user_tensor, targ_id, targ_idx, k):
        m = self._find_mask(user_tensor, targ_id)
        total_items_to_remove = self._process_sim_items(user_tensor, targ_id, targ_idx, user_tensor, m)
    
        return total_items_to_remove
    
    '''
        Computes similarity (mask) between user and items
        user_tensor: tensor representing the user
        targ_id: id of the target item
        returns: user/item similarity dictionary
    '''
    def _find_mask(self, user_tensor, targ_id):
        targ_embedd=self.item_embeddings[:,targ_id]
        item_sim_dict  ={} 
        for itm in user_tensor.nonzero().squeeze():
            user_emebdd=self.user_embeddings[:,itm]
            score=self._cosine_similarity_manual(targ_embedd,user_emebdd)
            item_sim_dict[itm]=score
        return item_sim_dict
    
    '''
        MORE DESCRIPTION HERE OF WHAT FUNCTION IS DOING 
        user_tensor: tensor representing the user
        targ_id: id of the target item
        targ_idx: index of the target item
        mask: user/item similarity dictionary
        k: how many places away target item must be
        returns: number of items to be removed
    '''
    def _process_sim_items(self, user_tensor, targ_id, targ_idx, mask, k=10):
        sorted_m = list(sorted(mask.items(), key=lambda item: item[1], reverse=True))
        total_items = 0

        for i in sorted_m: #i is never used -- should this be in a loop?
            total_items += 1
            p = self._mask_items(user_tensor, sorted_m, total_items)
            kw_dict=self.kw_dict #suggest directly using the target index instead of passing in the whole dictionary
            i1_rank = get_index_in_the_list(p, user_tensor, targ_id,self.recommender, **kw_dict) + 1
            
            if (i1_rank > targ_idx + k):
                return total_items

        return None
    
    '''
        Computes cosine similarity between two vectors manually.
        vec1: a vector
        vec2: another vector
        returns: cosine similarity between vec1 and vec2
    '''
    def _cosine_similarity_manual(self, vec1, vec2):
        norm_vec1 = np.linalg.norm(vec1)  
        norm_vec2 = np.linalg.norm(vec2)  
        
        # Avoid division by zero
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        dot_product = np.dot(vec1, vec2) 
        similarity = dot_product / (norm_vec1 * norm_vec2)
        return similarity
    
    '''
        MORE DESCRIPTION HERE OF WHAT FUNCTION IS DOING (one hot encoding? other?)
        user_tensor: tensor representing the user
        m1:
        p: 
        returns: mask of user_tensor
    '''
    def _mask_items(self,user_tensor, m1, p):
        mask = torch.zeros_like(user_tensor, dtype=torch.float32, device=self.device)
        indices = [int(item[0]) for item in m1[:p]]
        mask[indices] = 1

        return user_tensor - mask
