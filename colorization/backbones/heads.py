import torch
from torch import nn as nn


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, padding_mode='reflect')

    def forward(self, x):
        x = self.conv(x)
        x = nn.Tanh()(x)
        return x

    def initialize(self):
        nn.init.xavier_normal_(self.conv.weight, torch.nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.conv.bias)


class TanHActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(x)

    def initialize(self):
        pass
