import numpy as np
import os
from pathlib import Path
import pickle
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from collections import defaultdict
import torch
import torch.nn as nn
import optuna
import logging
import matplotlib.pyplot as plt
import wandb
from scripts.Config_Kw_Dict import get_kw_dict
from lxr_eval.src.lxr_eval.Evaluation import Evaluation
from lxr.src.lxr.Explainer import Explainer
from lxr.src.lxr.LXR_loss import LXR_loss
from scripts.load_data import targ_item, load_data
from scripts.recommender.recommenders_architecture import MLP, VAE













class LXR_10 ():

    def __init__(self, recommender_name,data_name, num_of_rand_users, task ):
        self.data_name=data_name
        self.recommender_name=recommender_name
        self.num_of_rand_users=num_of_rand_users
        self.kw=get_kw_dict()
        self.device=self.kw['device']
        self.recommender=self.load_recommender(recommender_name)
        self.task=task






    
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
        data_dict=load_data(self.data_name, self.recommender_name, self.kw)

        train_array, test_array, items_array=data_dict['train_array'], data_dict['test_array'], data_dict['items_array']
        targ_train, targ_test=targ_item(data_name=self.data_name, recommender_name=self.recommender_name, recommender=self.recommender,kw_dict=self.kw)
        random_rows = np.random.choice(test_array.shape[0], self.num_of_rand_users, replace=False)
        random_sampled_array = test_array[random_rows]

        
        ## loading hyperparameters
        param=self.kw['Predefined_hyperparameters'][self.recommender_name][self.data_name]
        learning_rate=param['learning_rate']
        lambda_neg=param['lambda_neg']
        lambda_pos=param['lambda_pos']
        alpha=param['alpha']
        batch_size=param['batch_size']
        explainer_hidden_size=param['explainer_hidden_size']
        epochs=param['epochs']


        ## wandb initialization
        wandb.init(
            project=f"{self.data_name}_{self.recommender_name}_LXR-10(reproducibility)training",
            config={
            'learning_rate' : learning_rate,
            'alpha' : alpha,
            'lambda_neg' : lambda_neg,
            'lambda_pos' : lambda_pos,
            'batch_size' : batch_size,
            'explainer_hidden_size' : explainer_hidden_size,
            'architecture' : 'LXR_combined',
            'activation_function' : 'Tanh',
            'loss_type' : 'logloss',
            'optimize_for' : 'Coverage and MPNR',
            'epochs':epochs
            })
        
        ## preparing data for training model
        loader = torch.utils.data.DataLoader(train_array, batch_size=batch_size, shuffle=True)
        num_batches = int(np.ceil(train_array.shape[0] / batch_size))


        coverage=[]
        
        ## A list for storing perturbations in all epochs. To select the best value among epochs
        MPRR_raw, MPRR_perc= [], []
        
        record_list=[] ## for storing all records for all epochs
        self.recommender.eval()

        num_items=self.kw['num_items'][self.data_name]
        num_features=num_items
        explainer = Explainer(num_features, num_items, explainer_hidden_size).to(self.device) 

        optimizer_comb = torch.optim.Adam(explainer.parameters(), learning_rate)
        kw_dict=self.kw
        loss_func = LXR_loss(self.data_name, self.recommender_name,lambda_pos, lambda_neg, alpha, self.recommender, kw_dict)

        print(f'======================== LXR10 ({self.task}task) run for {self.data_name} and {self.recommender_name}========================')
        
        for epoch in range(epochs):
            if epoch%15 == 0 and epoch>0: # decrease learning rate every 15 epochs
                learning_rate*= 0.1
                optimizer_comb.lr = learning_rate

            train_loss = 0 
            explainer.train()

            for batch_index, samples in enumerate(loader):

                # prepare data for explainer:
                user_tensors = torch.Tensor(samples[:,:-1]).to(self.device)
                user_ids = samples[:,-1]

                ### target items batch for training explainer (i1 is the target item id)
                i1=np.array([np.random.choice(targ_train[int(x)]) for x in user_ids])
                i1_vectors = items_array[i1]

                ### converting to a tensor
                i1_tensors = torch.Tensor(i1_vectors).to(self.device)
                n = user_tensors.shape[0]

                # zero grad:
                optimizer_comb.zero_grad()
                # forward:
                #### scoes for the target item
                m1 = explainer(user_tensors, i1_tensors)
            
                # caclulate loss
                comb_loss = loss_func(user_tensors, i1_tensors , i1, m1)

                train_loss += comb_loss*n
            
                # back propagation
                comb_loss.backward()
                optimizer_comb.step()

        

            torch.save(explainer.state_dict(), Path(kw_dict['checkpoints_path'], f'LXR10_{self.task}_{self.data_name}_{self.recommender_name}_{epoch}.pt'))
            
            explainer.eval()


            ## storing perturbations in each epoch(Raw and Percentage)
            MPRR_R, MPRR_P=[],[]
            records=[] ## for storing recprds for 1 epoch

            for j in range(self.num_of_rand_users):

                user_id = random_sampled_array[j][-1]
                user_tensor = torch.Tensor(random_sampled_array[j][:-1]).to(self.device)
                ## Target item for testing dataset
                i1,i1_index= self.select_target(user_id,targ_test)

                #i1 = np.random.choice(targ_test[user_id])
                #i1_index=list(targ_test[user_id]).index(i1)
                i1_vector = items_array[i1]
                i1_tensor = torch.Tensor(i1_vector).to(self.device)
                evaluation=Evaluation(self.data_name,self.recommender_name,explainer,self.recommender, kw_dict, k=10 )
                p,q= evaluation(user_tensor, i1,i1_index, i1_tensor)

                if p is not None:
                    MPRR_R.append(p)
                    MPRR_P.append(p/int(torch.sum(user_tensor)))

                    records.append({'MPNR':p,
                        'user_id': user_id,
                         'user_tensor': user_tensor,
                          'targ_item': i1,
                           'targ_index': i1_index,
                           'mask':q,
                            'items_array':items_array   })

            coverage.append(len(MPRR_R))  
        
            ## creating a list for storing values of "total_items_avg" in each epoch
            MPRR_raw.append(np.mean(MPRR_R))
            MPRR_perc.append(np.mean(MPRR_P))
            record_list.append(records)

            print(f'Finished epoch {epoch} with  MPRR(raw) {np.mean(MPRR_R)}, MPRR(%) {np.mean(MPRR_P)*100},'
            f'and Coverage (%) {len(MPRR_R)*100/self.num_of_rand_users}')
        
        

        print(f'Stop at trial with learning rate {learning_rate}, batch size={batch_size},' 
            f'explainer hidden size={explainer_hidden_size}, lambda_pos = {lambda_pos}, '
            f'lambda_neg = {lambda_neg}, alpha_parameter = {alpha}, '
            f'Best results at epoch {np.argmin(MPRR_raw)} with MPRR (raw) {np.min(MPRR_raw)},'
            f'MPRR (%)  {MPRR_perc[np.argmin(MPRR_raw)]*100}'
            f'and Coverage with value {coverage[np.argmin(MPRR_raw)]*100/self.num_of_rand_users}')  


        with open(Path(Path(os.getcwd(),'scripts'),f'checkpoints/Records_LXR10_{self.task}_{self.data_name}_{self.recommender_name}.pkl'), 'wb') as f:
                pickle.dump(record_list, f)

        return np.max(coverage)*100/ self.num_of_rand_users # return the best total items value in this trial


