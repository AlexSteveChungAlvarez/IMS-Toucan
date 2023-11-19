# Written by Ryan Li, 2021
# MIT License (https://choosealicense.com/licenses/mit/)
# Adapted by Alex Chung, 2023

import torch
from torch import nn

class InstanceNormalizationLayer(nn.Module):
    
    def __init__(self, **kwargs):
        super(InstanceNormalizationLayer,self).__init__(**kwargs)
        
    def _cal_mean_std(self, inputs, mask):
        expand_mask = mask.unsqueeze(2).type(inputs.dtype)
        sums = mask.sum(dim=-1,keepdim=True)
        
        mean = (inputs*expand_mask).sum(1) / sums
        
        std = torch.sqrt(
                ((inputs-mean.unsqueeze(1)).pow(2)).sum(1) / 
                    sums + 1e-05
              )
        
        return mean, std, expand_mask
    
    def forward(self, inputs, mask, return_mean_std=False):
        """
        inputs: [B, T, hidden_size]
        mask: [B, T]
        """
        mean, std, expand_mask = self._cal_mean_std(inputs, mask)
        
        outputs = (inputs - mean.unsqueeze(1)) /                   \
                    std.unsqueeze(1)*expand_mask
        
        if return_mean_std:
            return outputs, mean, std
        else:
            return outputs