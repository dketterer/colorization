import torch
from torch import nn as nn


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return nn.Tanh()(x)

    def initialize(self):
        nn.init.xavier_normal_(self.conv.weight, torch.nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.conv.bias)