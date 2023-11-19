# Written by Ryan Li, 2021
# MIT License (https://choosealicense.com/licenses/mit/)
# Adapted by Alex Chung, 2023

from torch import nn
from Layers.ConvModule import ConvModule

class EncConvBlock(nn.Module):
    def __init__(self, 
                 in_hidden_size,
                 enc_kernel_size, 
                 **kwargs):
        super(EncConvBlock,self).__init__(**kwargs)

        self.conv = ConvModule(
                    in_hidden_size,
                    enc_kernel_size)

    def forward(self, x):
        return x + self.conv(x)