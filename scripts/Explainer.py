import os
import torch
import torch.nn as nn

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
export_dir = os.getcwd()


'''
    CLASS DESCRIPTION
'''
class Explainer(nn.Module):
    def __init__(self, user_size, item_size, hidden_size):
        super(Explainer, self).__init__()
        self.users_fc = nn.Linear(in_features = user_size, out_features = hidden_size)
        self.items_fc = nn.Linear(in_features = item_size, out_features = hidden_size)
        self.bottleneck = nn.Sequential(
            nn.Tanh(),
            nn.Linear(in_features = hidden_size * 2, out_features = hidden_size),
            nn.Tanh(),
            nn.Linear(in_features = hidden_size, out_features = user_size),
            nn.Sigmoid()
        )
        
    '''
        DESCRIPTION OF METHOD
        user_tensor: tensor representing the user
        item_tensor: 
        returns: ???
    '''
    def forward(self, user_tensor, item_tensor):
        user_output = self.users_fc(user_tensor.float())
        item_output = self.items_fc(item_tensor.float())
        combined_output = torch.cat((user_output, item_output), dim=-1)
        expl_scores = self.bottleneck(combined_output)
        
        return expl_scores
