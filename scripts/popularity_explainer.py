import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from pathlib import Path
import torch
import torch.nn as nn
from os import path
from scripts.help_functions import Help_Functions
from scripts.recommender.recommenders_architecture import MLP, VAE
from scripts.Config_Kw_Dict import get_kw_dict
from scripts.load_data import load_data, targ_item
import pickle





class PopularityExplainer:
    def __init__ (self,recommender_name, data_name,task ):
            self.data_name=data_name
            self.recommender_name=recommender_name
            self.kw_dict=get_kw_dict()

            #self.pop_dict=self.kw_dict['pop_dict']
            self.device=self.kw_dict['device']
            #self.items_array=self.kw_dict['items_array']
            self.recommender=self.load_recommender(recommender_name)
            self.hf=Help_Functions(self.recommender, self.data_name, self.recommender_name,self.kw_dict)
            self.task=task





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
    

    def find_POP_mask(self, user_tensor):
        """" For finding masks based on popularity score """
        scores=self.Explainer(user_tensor)
        pop_x_masked = user_tensor* scores

        pop_item_sim_dict = {i: pop_x_masked[i].item() for i in range(len(pop_x_masked))}    

        return pop_item_sim_dict
        
    def Explainer(self, user_tensor):
        """ Explainer Scores"""
        score_tensor=torch.zeros_like(user_tensor)
        indx=torch.nonzero(user_tensor.squeeze())
        for i in indx:
            idx=i.item()
            score_tensor.squeeze()[idx]=self.pop_dict[idx].item()
        return score_tensor
    

    def mask_items(self, user_tensor, sim_items, total_items):

        mask = torch.zeros_like(user_tensor, dtype=torch.float32, device=self.device)
        indices = [item[0] for item in sim_items[:total_items]]
        mask[indices] = 1
        return user_tensor - mask
    


    def process_sim_items(self, sim_items, targ_id,targ_idx, user_tensor, user_hist_size,k):
       

        sorted_sim_items = list(sorted(sim_items.items(), key=lambda item: item[1], reverse=True))[:user_hist_size]

        total_items = 0
        for i in sorted_sim_items:
            total_items += 1
            ## masking user profile 
            POS_masked = self.mask_items(user_tensor, sorted_sim_items, total_items)
            ##index of first item and second item after masking
            kw_dict=self.kw_dict
            targ_rank = self.hf.get_index_in_the_list(POS_masked, user_tensor, targ_id) + 1
        
            if (targ_rank > k +targ_idx):
                return total_items
            

    
        return None


    def calculate_explanation(self,user_tensor,targ_id, targ_idx,user_hist_size,k ):
        pop_sim_items = self.find_POP_mask(user_tensor)
        total_items=self.process_sim_items(pop_sim_items, targ_id,targ_idx, user_tensor, user_hist_size,k)
        return total_items
    
    def select_target(self, user_id,targ_test):
        if self.task=="Top1":
            ## select Top1 item for Top1 task
            targ = int(targ_test[user_id][0])
            targ_indx=0
        else:
            ## Random sampling amonth Top10 items for Top10 task
            targ = np.random.choice(targ_test[user_id])
            targ_indx=list(targ_test[user_id]).index(targ)
        return targ, targ_indx
    
    def predict (self ):
        torch.manual_seed(42)
        np.random.seed(42)

        ## loading data
        dict_data=load_data(self.data_name, self.recommender_name, self.kw_dict)
        _, targ_test=targ_item( self.data_name, self.recommender_name, self.recommender, self.kw_dict)
        self.pop_dict=dict_data['pop_dict']
        # number of users for evaluations

        test_array=dict_data['test_array']
        num_of_rand_users = test_array.shape[0] 

        random_rows = np.random.choice(test_array.shape[0], num_of_rand_users, replace=False)
        random_sampled_array = test_array[random_rows]

        total_pert=[]  ## size of perturbations
        print(f'======================== Popularity Explainer ({self.task} Task) run for {self.data_name} and {self.recommender_name}========================')

        for j in range(num_of_rand_users):
            user_id = random_sampled_array[j][-1]
            user_tensor = torch.Tensor(random_sampled_array[j][:-1]).to(self.device)
            user_hist_size=int(torch.sum(user_tensor))
            ## For Top10 Evalaution
            targ_itm,targ_idx= self.select_target(user_id,targ_test)

            

            targ_vector = dict_data['items_array'][targ_itm]
            total_items= self.calculate_explanation(user_tensor, targ_itm, targ_idx, user_hist_size, k=10)
            if total_items is not None:
                total_pert.append(total_items)
            

        print(f'MPNR for Popularity Explainer ({self.task} Task) on {self.data_name} and {self.recommender_name} is ', np.mean(total_pert))
        print(f'Coverage for Popularity Explainer ({self.task} Task) on {self.data_name} and {self.recommender_name} is ', len(total_pert)*100/num_of_rand_users)
        
        with open(Path(Path(os.getcwd(),'scripts'),f'checkpoints/Records_POP_{self.task}_{self.data_name}_{self.recommender_name}.pkl'), 'wb') as f:
                pickle.dump(total_pert, f)