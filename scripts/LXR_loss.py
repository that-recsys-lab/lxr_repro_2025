
import pandas as pd
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
export_dir = os.getcwd()
from pathlib import Path
import torch
import torch.nn as nn
from help_functions import recommender_run




class LXR_loss(nn.Module):

    def __init__(self, lambda_pos, lambda_neg, alpha,recommender, output_type,kw_dict ):
        super(LXR_loss, self).__init__()
        
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        self.alpha = alpha
        self.recommender=recommender
        self.output_type=output_type
        self.kw_dict=kw_dict
        
    def forward(self, user_tensors, i1_tensors, i1_id, m1):

        
        neg_m1 = torch.sub(torch.ones_like(m1), m1)
        
        
        xm1_pos = user_tensors * m1
        
        

        xm1_neg = user_tensors * neg_m1
        

        if self.output_type=='single':
            
            ## Recommeder output for the target item by applying m1 mask (m1 is the mask for the target item)
            y1_m1_pos = torch.diag(recommender_run(xm1_pos, self.recommender, i1_tensors, item_id=i1_id, wanted_output = 'single', **self.kw_dict))

             ## Negative mask version for the target item
            y1_m1_neg = torch.diag(recommender_run(xm1_neg, self.recommender, i1_tensors, item_id=i1_id, wanted_output = 'single', **self.kw_dict))
        

        else:
            
            ### target item output
            y1_m1_pos = recommender_run(xm1_pos, self.recommender, i1_tensors, item_id=i1_id, wanted_output = 'vector', **self.kw_dict)
            y1_m1_neg = recommender_run(xm1_neg, self.recommender, i1_tensors, item_id=i1_id, wanted_output = 'vector', **self.kw_dict)


            
            rows1=torch.arange(len(i1_id))

            y1_m1_pos = y1_m1_pos[rows1, i1_id] 
            y1_m1_neg = y1_m1_neg[rows1, i1_id]    

        ## First loss term  ( maximizing rating score of the target item  for the positive and negative masks (same as LXR)   )
        pos_loss = - self.lambda_pos *torch.mean(torch.log(y1_m1_pos))
        neg_loss = self.lambda_neg * torch.mean(torch.log(y1_m1_neg))

       

        ## Third term ( sparsity terms )
        l = self.alpha * xm1_pos[user_tensors>0].mean() 



        
        ### combined loss (summing up all the terms
        combined_loss = pos_loss + neg_loss  + l        

        return combined_loss