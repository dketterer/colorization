import torch
from torch import nn
import torch.nn.functional as F

from colorization.backbones.residual_dense_block import RDB
from colorization.backbones.utils import register, PadToX
from colorization.backbones.resnet import ResNet
from torchvision.models import resnet as vrn


class RDBPixShuffle(nn.Module):
    def __init__(self, in_channels_1, in_channels_2):
        super(RDBPixShuffle, self).__init__()
        self.in_channels = in_channels_1 + in_channels_2
        self.up = nn.PixelShuffle(upscale_factor=2)

        self.conv_3x3 = nn.Conv2d(in_channels=in_channels_1, out_channels=in_channels_1,
                                  kernel_size=3,
                                  padding=1)
        self.rdb = RDB(n_channels=in_channels_1 // 4 + in_channels_2, nDenselayer=3,
                       growthRate=self.in_channels // 8)

    def forward(self, x1, x2):
        x1 = self.conv_3x3(x1)
        x1 = F.leaky_relu_(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.rdb(x)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                diffY // 2, diffY - diffY // 2])

        return x


class ResnetPixShuffle(nn.Module):
    def __init__(self, features, bilinear: bool = True):
        super(ResnetPixShuffle, self).__init__()
        self.features = features
        self.name = 'ResnetPixShuffle'

        is_light = self.features.bottleneck == vrn.BasicBlock
        channels = [64, 64, 128, 256, 512] if is_light else [64, 256, 512, 1024, 2048]
        self.base_channel_size = channels[0]

        self.smooth5 = nn.Sequential(
            RDB(n_channels=channels[4], nDenselayer=3, growthRate=channels[4] // 4),
            RDB(n_channels=channels[4], nDenselayer=3, growthRate=channels[4] // 4)
        )
        self.smooth4 = nn.Conv2d(channels[3], channels[3], kernel_size=1)
        self.smooth3 = nn.Conv2d(channels[2], channels[2], kernel_size=1)
        self.smooth2 = nn.Conv2d(channels[1], channels[1], kernel_size=1)
        self.smooth1 = nn.Conv2d(channels[0], channels[0], kernel_size=1)

        #                                                                       up + skip, out
        self.up1 = RDBPixShuffle(channels[4], channels[3])  # 2048, 1024 -> 512
        nChannels = channels[4] // 4 + channels[3]  # 1536
        self.up2 = RDBPixShuffle(nChannels, channels[2])  # 1536 + 512 -> 896
        nChannels = nChannels // 4 + channels[2]
        self.up3 = RDBPixShuffle(nChannels, channels[1])  # 896 + 256 -> 480
        nChannels = nChannels // 4 + channels[1]
        self.up4 = RDBPixShuffle(nChannels, channels[0])  # 480 + 64 -> 184
        nChannels = nChannels // 4 + channels[0]
        self.last_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(nChannels, channels[0], kernel_size=3, padding=1)
        )
        self.pad_to = PadToX(32)

    def forward(self, x):
        # need 3 channels for the pretrained resnet
        diffX, diffY, x, = self.pad_to(x)
        x = x.repeat(1, 3, 1, 1)
        c1, c2, c3, c4, c5 = self.features(x)

        c1 = self.smooth1(c1)
        c2 = self.smooth2(c2)
        c3 = self.smooth3(c3)
        c4 = self.smooth4(c4)
        c5 = self.smooth5(c5)

        x = self.up1(c5, c4)
        x = self.up2(x, c3)
        x = self.up3(x, c2)
        x = self.up4(x, c1)
        x = self.last_up(x)

        x = self.pad_to.remove_pad(x, diffX, diffY)
        return x

    def initialize(self):
        def init_layer(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

        self.apply(init_layer)

        self.features.initialize()


@register
def Resnet50PixShuffle():
    return ResnetPixShuffle(
        features=ResNet(layers=[3, 4, 6, 3], bottleneck=vrn.Bottleneck, url=vrn.model_urls['resnet50']))
