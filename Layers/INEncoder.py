# Written by Ryan Li, 2021
# MIT License (https://choosealicense.com/licenses/mit/)
# Adapted by Alex Chung, 2023

from torch import nn

from Layers.InstanceNormalizationLayer import InstanceNormalizationLayer
from Layers.EncConvBlock import EncConvBlock
from Layers.ConvNorm import ConvNorm
from Layers.LayerNorm import LayerNorm

class INEncoder(nn.Module):
    """
    Instance Normalization Encoder layer module

    Args:
        in_hidden_size (int):
        out_hidden_size (int):
        n_conv_blocks (int):
        enc_kernel_size (int):
        layer_norm_eps (float):
    """
    
    def __init__(self,
                 input_size,
                 in_hidden_size,
                 out_hidden_size,
                 n_conv_blocks,
                 enc_kernel_size,
                 **kwargs):
        super(INEncoder,self).__init__(**kwargs)
        self.layer_norm = LayerNorm(in_hidden_size)
        self.in_conv = ConvNorm(input_size,in_hidden_size)
        self.out_conv = ConvNorm(in_hidden_size,out_hidden_size)
        self.inorm = InstanceNormalizationLayer()
        self.conv_blocks = nn.ModuleList(
            [EncConvBlock(in_hidden_size,in_hidden_size,enc_kernel_size)
                            for _ in range(n_conv_blocks)]
                        )

    def forward(self,x,mask):
        means = []
        stds = []
        x = x.transpose(1, 2) # num_mels (80) need to be in the channels axis (1).
        y = self.in_conv(x) # 80 -> 256

        for block in self.conv_blocks:
            y = block(y)
            y = self.layer_norm(y.transpose(1,2))
            y, mean, std = self.inorm(y, mask, return_mean_std=True)
            y = self.layer_norm(y).transpose(1,2)
            means.append(mean)
            stds.append(std)

        y = self.out_conv(y).transpose(1, 2) # 256 -> 192
        means.reverse()
        stds.reverse()

        return y, means, stds