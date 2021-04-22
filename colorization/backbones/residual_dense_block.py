# Residual dense block (RDB) architecture
import torch
from torch import nn
from torch.nn import functional as F

from colorization.backbones.unetparts import UpPad


class Dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(Dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        out = F.leaky_relu_(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class RDB(nn.Module):
    """https://arxiv.org/abs/1802.08797"""

    def __init__(self, n_channels=64, nDenselayer=3, growthRate=32):
        super(RDB, self).__init__()
        nChannels_ = n_channels
        modules = []
        for i in range(nDenselayer):
            modules.append(Dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, n_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


class RDBUp(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, out_channels):
        super(RDBUp, self).__init__()
        self.in_channels = in_channels_1 + in_channels_2
        self.out_channels = out_channels

        self.up = UpPad()

        self.conv_1x1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1,
                                  bias=False)
        self.rdb = RDB(n_channels=self.out_channels, nDenselayer=3, growthRate=self.out_channels // 2)

    def forward(self, x1, x2):
        # need x2 just for the size
        x1 = self.up(x1, x2)

        x = torch.cat([x2, x1], dim=1)
        x = self.conv_1x1(x)
        x = F.leaky_relu_(x)
        x = self.rdb(x)
        return x
