import logging
import numpy as np
import optuna
import os
import torch
from pathlib import Path

from help_functions import sample_indices, recommender_evaluations
from load_data import load_data
from recommenders_architecture import MLP, VAE

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
export_dir = os.getcwd()

'''
    CLASS DESCRIPTION
    Note: I'm not going to bother deleting all these commented out lines of code
    in case they are still helpful for testing. But should be deleted before
    the code is finalized
'''
class RecommenderTrainer:
    def __init__(self, data_name, recommender_name, **kw_dict):
        self.data_name = data_name
        self.recommender_name = recommender_name
        
        #self.output_type_dict = {"VAE": "multiple", "MLP": "single"}
        #self.num_users_dict = {"ML1M": 6037, "Yahoo": 13797, "Pinterest": 19155}
        #self.num_items_dict = {"ML1M": 3381, "Yahoo": 4604, "Pinterest": 9362}

        #self.output_type = self.output_type_dict[recommender_name]
        #self.num_users = self.num_users_dict[data_name]
        #self.num_items = self.num_items_dict[data_name]

        '''
        self.VAE_config = {
            "enc_dims": [256, 64],
            "dropout": 0.5,
            "anneal_cap": 0.2,
            "total_anneal_steps": 200000
        }
        '''

        #self.data = load_data(data_name, recommender_name)
        self.kw_dict = kw_dict
        '''
        self.kw_dict = {
            'device': torch.device("mps" if torch.cuda.is_available() else "cpu"),
            'num_items': self.num_items,
            'pop_array': self.data['pop_array'],
            'all_items_tensor': self.data['all_items_tensor'],
            'static_test_data': self.data['static_test_data'],
            'items_array': self.data['items_array'],
            'output_type': self.output_type,
            'recommender_name': self.recommender_name
        }
        '''
        #self.checkpoints_path = Path("checkpoints")
        #self.checkpoints_path.mkdir(parents=True, exist_ok=True)
        #self.train_losses_dict = {}
        #self.test_losses_dict = {}
        #self.HR10_dict = {}

    '''
    '''
    def optimize(self, n_trials=20):
        objective_fn = self.mlp_objective if self.recommender_name == "MLP" else self.vae_objective
        study = optuna.create_study(direction='maximize')
        self.logger.info("Start optimization.")
        study.optimize(objective_fn, n_trials=n_trials)

    def MLP_objective(self,trial):
        lr = trial.suggest_float('learning_rate', 0.001, 0.01)
        batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
        beta = trial.suggest_float('beta', 0, 4) # hyperparameter that weights the different loss terms
        epochs = 10
        model = MLP(hidden_dim, self.kw_dict)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_losses = []
        test_losses = []
        train_losses_dict = {}
        test_losses_dict={}
        HR10_dict={}
        hr10 = []
        
        print(f'======================== new run - {self.recommender_name} ========================')
        
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.FileHandler(f"{self.recommender_name}_{self.data_name}_Optuna.log", mode="w"))
        optuna.logging.enable_propagation()
        optuna.logging.disable_default_handler()
        logger.info(f'======================== new run - {self.recommender_name} ========================')

        device=self.kw_dict['device']

        data=load_data(self.data_name, self.recommender_name)
        num_training = data['train_data'].shape[0]
        num_batches = int(np.ceil(num_training / batch_size))

        for epoch in range(epochs):
            train_matrix = sample_indices(data['train_data'].copy(), **self.kw_dict)
            perm = np.random.permutation(num_training)
            loss = []
            train_pos_loss=[]
            train_neg_loss=[]
            if epoch!=0 and epoch%10 == 0: # decrease the learning rate every 10 epochs
                lr = 0.1*lr
                optimizer.lr = lr
            
            for b in range(num_batches):
                optimizer.zero_grad()
                if (b + 1) * batch_size >= num_training:
                    batch_idx = perm[b * batch_size:]
                else:
                    batch_idx = perm[b * batch_size: (b + 1) * batch_size]    
                batch_matrix = torch.FloatTensor(train_matrix[batch_idx,:-2]).to(device)

                batch_pos_idx = train_matrix[batch_idx,-2]
                batch_neg_idx = train_matrix[batch_idx,-1]
                
                batch_pos_items = torch.Tensor(data['items_array'][batch_pos_idx]).to(device)
                batch_neg_items = torch.Tensor(data['items_array'][batch_neg_idx]).to(device)
                
                pos_output = torch.diagonal(model(batch_matrix, batch_pos_items))
                neg_output = torch.diagonal(model(batch_matrix, batch_neg_items))
                
                # MSE loss
                pos_loss = torch.mean((torch.ones_like(pos_output)-pos_output)**2)
                neg_loss = torch.mean((neg_output)**2)
                
                batch_loss = pos_loss + beta*neg_loss
                batch_loss.backward()
                optimizer.step()
                
                loss.append(batch_loss.item())
                train_pos_loss.append(pos_loss.item())
                train_neg_loss.append(neg_loss.item())
                
            print(f'train pos_loss = {np.mean(train_pos_loss)}, neg_loss = {np.mean(train_neg_loss)}')    
            train_losses.append(np.mean(loss))
            torch.save(model.state_dict(), Path(self.checkpoints_path, f'MLP_{data_name}_{round(lr,4)}_{batch_size}_{trial.number}_{epoch}.pt'))

            model.eval()
            test_matrix = np.array(data['static_test_data'])
            test_tensor = torch.Tensor(test_matrix[:,:-2]).to(device)
            
            test_pos = test_matrix[:,-2]
            test_neg = test_matrix[:,-1]
            
            row_indices = np.arange(test_matrix.shape[0])
            test_tensor[row_indices,test_pos] = 0
            
            pos_items = torch.Tensor(data['items_array'][test_pos]).to(device)
            neg_items = torch.Tensor(data['items_array'][test_neg]).to(device)
            
            pos_output = torch.diagonal(model(test_tensor, pos_items).to(device))
            neg_output = torch.diagonal(model(test_tensor, neg_items).to(device))
            
            pos_loss = torch.mean((torch.ones_like(pos_output)-pos_output)**2)
            neg_loss = torch.mean((neg_output)**2)
            print(f'test pos_loss = {pos_loss}, neg_loss = {neg_loss}')
            
            hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR = recommender_evaluations(model, **self.kw_dict)
            hr10.append(hit_rate_at_10) # metric for monitoring
            print(hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR)
            
            test_losses.append(-hit_rate_at_10)
            if epoch>5: # early stop if the HR@10 decreases for 4 epochs in a row
                if test_losses[-2]<=test_losses[-1] and test_losses[-3]<=test_losses[-2] and test_losses[-4]<=test_losses[-3]:
                    logger.info(f'Early stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {np.argmin(test_losses)} with value {np.min(test_losses)}')
                    train_losses_dict[trial.number] = train_losses
                    test_losses_dict[trial.number] = test_losses
                    HR10_dict[trial.number] = hr10
                    return max(hr10)
                
        logger.info(f'Stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {np.argmin(test_losses)} with value {np.min(test_losses)}')
        train_losses_dict[trial.number] = train_losses
        test_losses_dict[trial.number] = test_losses
        HR10_dict[trial.number] = hr10
        return max(hr10)


    def vae_objective(self, trial):
        lr = trial.suggest_float('learning_rate', 0.001, 0.01)
        batch_size = trial.suggest_categorical('batch_size', [64,128,256])
        epochs = 20
        model = VAE(VAE_config, self.kw_dict) #where is VAE_config coming from?
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_losses = []
        test_losses = []
        hr10 = []
        print('======================== new run ========================')
        logger.info('======================== new run ========================')
        
        for epoch in range(epochs):
            if epoch!=0 and epoch%10 == 0:
                lr = 0.1*lr
                optimizer.lr = lr
            loss = model.train_one_epoch(train_array, optimizer, batch_size)
            train_losses.append(loss)
            torch.save(model.state_dict(), Path(checkpoints_path, f'VAE_{data_name}_{trial.number}_{epoch}_{round(lr,4)}_{batch_size}.pt'))


            model.eval()
            test_matrix = static_test_data.to_numpy()
            test_tensor = torch.Tensor(test_matrix[:,:-2]).to(device)
            test_pos = test_array[:,-2]
            test_neg = test_array[:,-1]
            row_indices = np.arange(test_matrix.shape[0])
            test_tensor[row_indices,test_pos] = 0
            output = model(test_tensor).to(device)
            print('output:',output.shape)
            pos_loss = -output[row_indices,test_pos].mean()
            neg_loss = output[row_indices,test_neg].mean()
            print(f'pos_loss = {pos_loss}, neg_loss = {neg_loss}')
            
            hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR = recommender_evaluations(model, **kw_dict)
            hr10.append(hit_rate_at_10)
            print(hit_rate_at_10, hit_rate_at_50, hit_rate_at_100, MRR, MPR)
            
            test_losses.append(pos_loss.item())
            if epoch>5:
                if test_losses[-2]<test_losses[-1] and test_losses[-3]<test_losses[-2] and test_losses[-4]<test_losses[-3]:
                    logger.info(f'Early stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {np.argmin(test_losses)} with value {np.min(test_losses)}')
                    train_losses_dict[trial.number] = train_losses
                    test_losses_dict[trial.number] = test_losses
                    HR10_dict[trial.number] = hr10
                    return max(hr10)
        
        logger.info(f'Stop at trial with batch size = {batch_size} and lr = {lr}. Best results at epoch {np.argmin(test_losses)} with value {np.min(test_losses)}')
        train_losses_dict[trial.number] = train_losses
        test_losses_dict[trial.number] = test_losses
        HR10_dict[trial.number] = hr10
        return max(hr10)
















































