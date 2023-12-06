# Written by KimythAnly, 2020
# MIT License (https://choosealicense.com/licenses/mit/)
# Adapted by Alex Chung, 2023

from torch import nn

class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding='same', dilation=1, groups=1, bias=True, initializer_range=0.02, padding_mode='zeros'):
        super(ConvNorm, self).__init__()
       
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              groups=groups,
                              bias=bias, padding_mode=padding_mode)

        nn.init.trunc_normal_(
            self.conv.weight, std=initializer_range)

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal