# Written by Ryan Li, 2021
# MIT License (https://choosealicense.com/licenses/mit/)
# Adapted by Alex Chung, 2023

from torch import nn
from Layers.ConvModule import ConvModule

class DecConvBlock(nn.Module):
    def __init__(self, 
                 input_size,
                 in_hidden_size,
                 dec_kernel_size,
                 gen_kernel_size,
                 **kwargs):
        super(DecConvBlock,self).__init__(**kwargs)

        self.dec_conv = ConvModule(
                    input_size,
                    in_hidden_size,
                    dec_kernel_size)
        self.gen_conv = ConvModule(
                    input_size,
                    in_hidden_size,
                    gen_kernel_size)
        
    def forward(self, x):
        y = self.dec_conv(x)
        y = y + self.gen_conv(y)
        return x + y