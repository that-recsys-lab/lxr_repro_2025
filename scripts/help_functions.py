import numpy as np
import torch

from load_data import load_data

'''
    CLASS DESCRIPTION
    Note: typically a helper function file doesn't have a class object, just loose functions.
    Not super picky about this though

    HOWEVER -- if you keep it as a class you will have to change your import statements.
    Instead of:
    from help_functions import get_index_in_list

    it would be:
    from help_functions import Help_Functions
    then in code:
    help_functions_class = Help_Functions(....)
    help_functions_class.get_index_in_list

    which I think is also why helper functions usually aren't part of a class.

    I thought about changing this to not be a class myself, but want to discuss more. It also involves more reactoring here
    (and possibly elsehwere where this code is called) because of how some methods use fields
'''
class Help_Functions:
    def __init__(self, recommender, data_name, recommender_name, kw):
        self.recommender = recommender

        dict_data = load_data(data_name, recommender_name, kw)
        self.all_items_tensor = dict_data['all_items_tensor']
        self.pop_array = dict_data['pop_array']
        self.static_test_data = dict_data['static_test_data']

        self.device = kw['device'] # I don't remember off the top of my head but will this break if kw doesn't contain 'device'? 
                                    # one fix is wherever the entry point of the program is, to evaluate the keywords and if it doesn't
                                    # have some of these terms set kw['device'] = None or throw an error or something
        self.num_items = kw['num_items']
    
    '''
        DESCRIPTION OF METHOD
        data: ???
        returns: ???
    '''
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
    
    '''
        DESCRIPTION OF METHOD
        user_tensor: tensor representing the user
        original_user_tensor: ???
        item_id: ???
        returns: ???
    '''
    def get_index_in_list(self, user_tensor, original_user_tensor, item_id):
        top_k_list = list(self.get_top_k(user_tensor, original_user_tensor).keys())
        return top_k_list.index(item_id)
    
    '''
        DESCRIPTION OF METHOD
        user_tensor: tensor representing the user
        original_user_tensor: ???
        returns: ???

    '''
    def get_top_k(self, user_tensor, original_user_tensor):
        item_prob_dict = {}
        output_model = self.recommender_run(user_tensor, self.all_items_tensor, wanted_output='vector').cpu().detach().numpy().tolist()
        original_user_vector = original_user_tensor.cpu().numpy()[:self.num_items]
        catalog = np.ones_like(original_user_vector) - original_user_vector
        output = catalog * output_model

        for i in range(len(output)):
            if catalog[i] > 0:
                item_prob_dict[i] = output[i]

        return dict(sorted(item_prob_dict.items(), key=lambda item: item[1], reverse=True))

    '''
        DESCRIPTION OF METHOD
        user_tensor: tensor representing the user
        item_tensor: ???
        item_id: ???
        wanted_output: ???
        output_type: ???
        returns: ???
    '''
    def recommender_run(self, user_tensor, item_tensor=None, item_id=None, wanted_output='single', output_type='single'):
        if output_type == 'single':
            out = self.recommender(user_tensor, item_tensor)
        else:
            out = self.recommender(user_tensor)

        if wanted_output == 'single':
            if output_type == 'single':
                return out
            else:
                return out.squeeze()[item_id]
        return out.squeeze()

    '''
        DESCRIPTION OF METHOD
        user_tensor: tensor representing the user
        returns: ???
    '''
    def get_user_recommended_item(self, user_tensor):
        user_res = self.recommender_run(user_tensor, self.all_items_tensor, wanted_output='vector')[:self.num_items]
        user_tensor = user_tensor[:self.num_items]
        catalog = torch.ones_like(user_tensor) - user_tensor
        recommendations = torch.mul(user_res, catalog)
        sorted_recommendations = torch.argsort(recommendations, descending=True)
        return sorted_recommendations[:10]

    '''
        DESCRIPTION OF METHOD
        ranked_list: ???
        target_item: ???
        returns: ???

        NOTE: not used anywhere; delete if unused
    '''
    def get_ndcg(self, ranked_list, target_item):
        if target_item not in ranked_list:
            return 0.0
        target_idx = torch.tensor(ranked_list.index(target_item), device=self.device)
        dcg = torch.reciprocal(torch.log2(target_idx + 2))
        return dcg.item()

    '''
        DESCRIPTION OF METHOD
        returns: ???
    '''
    def recommender_evaluations(self):
        counter_10 = counter_50 = counter_100 = RR = PR = n =  0 

        for entry in self.static_test_data:
            item_id = entry[-2]
            user_tensor = torch.tensor(entry[:-2], device=self.device)
            user_tensor[item_id] = 0

            index = self.get_index_in_list(user_tensor, user_tensor, item_id) + 1
            if index <= 10: counter_10 += 1
            if index <= 50: counter_50 += 1
            if index <= 100: counter_100 += 1 #are these supposed to be elif? Or is it desired behavior that for index = 5, each counter gets + 1
            RR += 1 / index
            PR += index / self.num_items
            n += 1

        return (
            counter_10 / n,
            counter_50 / n,
            counter_100 / n,
            RR / n,
            100 * PR / n,
        )
