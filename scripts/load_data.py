import numpy as np
import os
import pandas as pd
import pickle
import torch
from pathlib import Path

from help_functions import Help_Functions

TARGET_LIST_SIZE = 10

'''
    Load and preprocess training and test data for a given dataset.

    Args:
        data_name (str): Name of the dataset (e.g., 'ML1M').
        recommender_name (str): Name of the recommender model (unused here but kept for compatibility).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Processed train and test data arrays.
'''
def load_data(data_name, recommender_name, kw_Dict): #I understand recommender_name is being ept for compatibility BUT it is easier to read if unused parameters are deleted...
    DP_DIR = Path("processed_data", data_name) 
    export_dir = Path(os.getcwd())
    files_path = Path(export_dir, DP_DIR)
    num_items = kw_Dict['num_items']['data_name']

    if num_items is None:
        raise ValueError(f"Unknown dataset: {data_name}")

    # Load data
    train_data = pd.read_csv(Path(files_path,f'train_data_{data_name}.csv'), index_col=0)
    test_data = pd.read_csv(Path(files_path,f'test_data_{data_name}.csv'), index_col=0)
    train_data['user_id'] = train_data.index
    test_data['user_id'] = test_data.index
    static_test_data = pd.read_csv(Path(files_path,f'static_test_data_{data_name}.csv'), index_col=0)

    with open(Path(files_path,f'pop_dict_{data_name}.pkl'), 'rb') as f:
        pop_dict = pickle.load(f)

    train_array = train_data.to_numpy()
    test_array = test_data.to_numpy()
    items_array = np.eye(num_items)
    
    all_items_tensor = torch.Tensor(items_array).to(kw_Dict['device'])
    
    # Vectorized in-place removal: set target item column to 0 in static_test_data
    target_indices = static_test_data.iloc[:, -2].astype(int).values
    rows = np.arange(static_test_data.shape[0])
    static_test_data.iloc[rows, target_indices] = 0

    pop_array = np.zeros(len(pop_dict))
    for key, value in pop_dict.items():
        pop_array[key] = value

    return {'train_array': train_array,
         'test_array': test_array,
         'train_data': train_data,
         'test_data': test_data,
         'pop_array': pop_array,
         'items_array': items_array,
         'static_test_data': static_test_data,
         'all_items_tensor': all_items_tensor
    }

'''
    Load / create top recommended items dict
    data_name:
    recommender_name:
    recommender:
    kw_dict:
    returns:

    NOTE: function appears to be unused; delete if unused
    Did leave some comments but didn't redo this function as much as I would've,
    because it might just get deleted
'''
def targ_item(data_name, recommender_name, recommender, kw_dict):
    base_path = Path("processed_data") / data_name
    full_path = Path(os.getcwd()) / base_path
    device = kw_dict['device']
    dic = load_data(data_name, recommender_name, kw_dict)
    hf = Help_Functions(recommender, data_name, recommender_name, kw_dict)

    train_array = dic['train_array']
    test_array = dic['test_array']

    ## target item for training and testing
    targ_train, targ_test = {}, {}
    
    for i in range(train_array.shape[0]): #suggest making this chunk (for loop + writing to file) a function as it's the same for train/test
        user_index = train_array[i][-1]
        user_tensor = torch.Tensor(train_array[i][:-1]).to(device)
        recomm_list = hf.get_user_recommended_item(user_tensor)
        ## Sampling for the target item 
        targ_train[user_index] = np.array(recomm_list[0:TARGET_LIST_SIZE].cpu())
    for i in range(test_array.shape[0]): #instead do for item in test_array?
        user_index = test_array[i][-1]
        user_tensor = torch.Tensor(test_array[i][:-1]).to(device)
        recomm_list = hf.get_user_recommended_item(user_tensor)
        ## Sampling for the target item 
        targ_test[user_index] = np.array(recomm_list[0:TARGET_LIST_SIZE].cpu())

    with open(Path(full_path,f'targ_train_{data_name}_{recommender_name}.pkl'), 'wb') as f:
        pickle.dump(targ_train, f)
    
    with open(Path(full_path,f'targ_test_{data_name}_{recommender_name}.pkl'), 'wb') as f:
        pickle.dump(targ_test, f)
    
    return targ_train, targ_test