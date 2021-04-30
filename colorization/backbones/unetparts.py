import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck

""" Parts of the U-Net model """


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, fix_bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if not mid_channels:
            mid_channels = self.out_channels
        self.double_conv = nn.Sequential(
            # bias should be False, but trained models with bias exist...
            # this is because of the following BN the offset is annihilated
            nn.Conv2d(self.in_channels, mid_channels, kernel_size=3, padding=1, bias=not fix_bias),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # bias should be False, but trained models with bias exist...
            nn.Conv2d(mid_channels, self.out_channels, kernel_size=3, padding=1, bias=not fix_bias),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Upv2(Up):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__(in_channels, out_channels, bilinear)

        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.conv = nn.Sequential(Bottleneck(in_channels, out_channels // 4, downsample=downsample),
                                  Bottleneck(out_channels, out_channels // 4),
                                  Bottleneck(out_channels, out_channels // 4))


class UpPad(nn.Module):
    def __init__(self):
        super(UpPad, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if diffX > 0 or diffY > 0:
            # Hack for https://github.com/pytorch/pytorch/issues/13058
            with autocast(enabled=False):
                x1 = x1.float()

                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])

        return x1
