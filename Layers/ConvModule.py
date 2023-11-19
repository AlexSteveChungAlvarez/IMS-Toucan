# Written by Ryan Li, 2021
# MIT License (https://choosealicense.com/licenses/mit/)
# Adapted by Alex Chung, 2023

from torch import nn
from Layers.ConvNorm import ConvNorm

class ConvModule(nn.Module):
    def __init__(self, filter_size, kernel_size, **kwargs):
        super(ConvModule,self).__init__(**kwargs)
        
        self.seq = nn.Sequential(
                    ConvNorm(filter_size,kernel_size),
                    nn.LazyBatchNorm1d(),
                    nn.LeakyReLU(),
                    ConvNorm(filter_size,kernel_size)
                    )
    def forward(self,x):
        return self.seq(x)