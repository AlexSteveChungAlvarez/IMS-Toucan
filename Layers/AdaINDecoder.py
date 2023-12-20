# Written by Ryan Li, 2021
# MIT License (https://choosealicense.com/licenses/mit/)
# Adapted by Alex Chung, 2023

from torch import nn

from Layers.InstanceNormalizationLayer import InstanceNormalizationLayer
from Layers.DecConvBlock import DecConvBlock
from Layers.ConvNorm import ConvNorm
from Layers.LayerNorm import LayerNorm

class AdaINDecoder(nn.Module):
    """
    Instance Normalization Decoder layer module

    Args:
        in_hidden_size (int):
        out_hidden_size (int):
        n_conv_blocks (int):
        dec_kernel_size (int):
        gen_kernel_size (int):
        layer_norm_eps (float):
    """
    
    def __init__(self,
                 input_size,
                 in_hidden_size,
                 out_hidden_size,
                 n_conv_blocks,
                 dec_kernel_size,
                 gen_kernel_size,
                 **kwargs):
        super(AdaINDecoder,self).__init__(**kwargs)

        self.in_conv = ConvNorm(input_size,in_hidden_size)
        self.out_conv = ConvNorm(in_hidden_size,out_hidden_size)
        self.inorm = InstanceNormalizationLayer()
        self.conv_blocks = nn.ModuleList(
                    [DecConvBlock(in_hidden_size,
                                 in_hidden_size,
                                 dec_kernel_size,
                                 gen_kernel_size)
                    for _ in range(n_conv_blocks)]
                    )
        self.layer_norm = LayerNorm(in_hidden_size)
        
    def forward(self,enc,cond,mask):
        _, means, stds = cond
        enc = enc.transpose(1, 2)
        y = self.in_conv(enc) # 192 -> 256
        for block,mean,std in zip(self.conv_blocks,means,stds):
            y = self.layer_norm(y.transpose(1,2))
            y = self.inorm(y, mask)
            y = y * std.unsqueeze(1) + mean.unsqueeze(1)
            y = self.layer_norm(y).transpose(1,2)
            y = block(y)

        y = self.out_conv(y).transpose(1, 2) # 256 -> 80

        return y