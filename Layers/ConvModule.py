# Written by Ryan Li, 2021
# MIT License (https://choosealicense.com/licenses/mit/)
# Adapted by Alex Chung, 2023

from torch import nn
from Layers.ConvNorm import ConvNorm

class ConvModule(nn.Module):
    def __init__(self, input_size, filter_size, kernel_size, **kwargs):
        super(ConvModule,self).__init__(**kwargs)
        
        self.seq = nn.Sequential(
                    ConvNorm(input_size,filter_size,kernel_size),
                    nn.BatchNorm1d(filter_size),
                    nn.LeakyReLU(),
                    ConvNorm(filter_size,input_size,kernel_size)
                    )
    def forward(self,x):
        return self.seq(x)