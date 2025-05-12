import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from scripts.help_functions import Help_Functions
from pathlib import Path
from scripts.load_data import load_data, targ_item
from scripts.recommender.recommenders_architecture import MLP, VAE
from scripts.Config_Kw_Dict import get_kw_dict
import pickle





class LatentFactorExplainer:
    
    def __init__(self, recommender_name, data_name):
        self.kw=get_kw_dict()
        self.data_name=data_name
        self.recommender_name=recommender_name
        MF_recommender=self.load_recommender(recommender_name='MLP')
        recommender=self.load_recommender(recommender_name)
        self.user_embeddings = MF_recommender.users_fc.weight.detach().cpu().numpy()
        self.item_embeddings = MF_recommender.items_fc.weight.detach().cpu().numpy()
        self.recommender=recommender
        self.device=self.kw['device']
        self.hf=Help_Functions(self.recommender, self.data_name, self.recommender_name,self.kw)



    
         


    def load_recommender(self,recommender_name):
        kw_dict= self.kw
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
        
    



    def find_mask(self, user_tensor, targ_id):
        """ computing mask for each user """

        targ_embedd=self.item_embeddings[:,targ_id]
        item_sim_dict  ={} 
        for itm in user_tensor.nonzero().squeeze():
            user_emebdd=self.user_embeddings[:,itm]
            score=self.cosine_similarity_manual(targ_embedd,user_emebdd)
            item_sim_dict[itm]=score
        return item_sim_dict
    


    def cosine_similarity_manual(self,vec1, vec2):
        """Compute cosine similarity between two vectors manually."""
        
        dot_product = np.dot(vec1, vec2)  # Compute dot product
        norm_vec1 = np.linalg.norm(vec1)  # Compute L2 norm of vec1
        norm_vec2 = np.linalg.norm(vec2)  # Compute L2 norm of vec2
        
        # Avoid division by zero
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        similarity = dot_product / (norm_vec1 * norm_vec2)
        return similarity
    
    
    
    
    def mask_items(self,user_tensor, m1, p):
        # Helper function to mask items
       
        mask = torch.zeros_like(user_tensor, dtype=torch.float32, device=self.device)
        indices = [int(item[0]) for item in m1[:p]]
        mask[indices] = 1

        return user_tensor - mask
    


    def process_sim_items(self, mask, targ_id, targ_idx, user_tensor, k):


        sorted_m = list(sorted(mask.items(), key=lambda item: item[1], reverse=True))

        total_items = 0 ## total perturbation

        for i in sorted_m:

            total_items += 1
            p = self.mask_items(user_tensor, sorted_m, total_items)
            
            ##index of target item 
            i1_rank = self.hf.get_index_in_the_list(p, user_tensor, targ_id) + 1
            
            if (i1_rank > targ_idx + k):
                #return total_items, i, Indic_score
                return total_items, sorted_m


    
        return None, None
    

    
    def Calculate_Explanation (self, user_tensor, targ_id, targ_idx, k):

        # Find mask
        m = self.find_mask(
            user_tensor, targ_id)
        #### total items to be removed from the users profile 
        total_items, perturb=self.process_sim_items(
            m, targ_id, targ_idx, user_tensor, k=10)
    

        return total_items, perturb
    




    def predict(self):

        torch.manual_seed(42)
        np.random.seed(42)

        MPNR_Row=[]
        records_list=[]
        ## loading train9ng and test datasets
        dict_data=load_data(self.data_name, self.recommender_name, self.kw )
        items_array=dict_data['items_array']
        ## loading train and test targ items
        _,targ_test=targ_item(self.data_name, self.recommender_name, self.recommender, self.kw)

        test_array=dict_data['test_array']
        num_of_rand_users = test_array.shape[0]   # number of users for evaluations 
        random_rows = np.random.choice(test_array.shape[0], num_of_rand_users, replace=False)
        random_sampled_array = test_array[random_rows]
        print(f'======================== Latent Factors (LF) Explainer run for {self.data_name} and {self.recommender_name}========================')


        for j in range(random_sampled_array.shape[0]):
            
            user_id = random_sampled_array[j][-1]
            user_tensor = torch.Tensor(random_sampled_array[j][:-1]).to(self.device)
            targ = np.random.choice(targ_test[user_id])
            targ_indx=list(targ_test[user_id]).index(targ)
            targ_vector = items_array[targ]

            p,q = self.Calculate_Explanation (user_tensor, targ , targ_indx, k=10) ## P is perturbation size and q is sorted m

            if p is not None:
                record={'MPNR':p,
                        'NPNR_P':p/int(torch.sum(user_tensor)), 
                        'user_id': user_id,
                         'user_tensor': user_tensor,
                          'targ_item': targ,
                           'targ_index': targ_indx,
                           'mask':q,
                            'items_array':items_array   } ## Saving these files for ploting figures
                records_list.append(record)
                MPNR_Row.append(p)
                

        print(f'MPNR Row for latent factors similarity for {self.data_name} and {self.recommender_name}:', np.mean(MPNR_Row))
        print(f'Coverage for latent factors similarity for {self.data_name} and {self.recommender_name}:', len(MPNR_Row)*100/num_of_rand_users)

        with open(Path(Path(os.getcwd(),'scripts'),f'checkpoints/Records_LF_{self.data_name}_{self.recommender_name}.pkl'), 'wb') as f:
                pickle.dump(records_list, f)
    
        



