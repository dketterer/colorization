import torch
from torch import nn
from torch.cuda.amp import autocast
from torchvision.models import densenet
import torch.nn.functional as F
from torchvision.models.densenet import _DenseBlock, _Transition

from colorization.backbones.densenet import DenseNet
from colorization.backbones.utils import register, PadToX
from colorization.backbones.residual_dense_block import RDB
from colorization.backbones.unetparts import UpPad
from colorization.backbones.residual_dense_block import RDBUp


class DenseNetUNetRDB(nn.Module):
    def __init__(self, features):
        super(DenseNetUNetRDB, self).__init__()
        self.features = features
        self.name = 'DenseUnetRDB'

        channels = [64, 256, 512, 1024, 1024]
        self.base_channel_size = channels[0]

        self.smooth5 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels[4], channels[4], kernel_size=1),
            nn.ReLU(),
            RDB(n_channels=channels[4], nDenselayer=3, growthRate=channels[4] // 4)

        )
        self.smooth4 = nn.Sequential(nn.ReLU(),
                                     nn.Conv2d(channels[3], channels[3], kernel_size=1),
                                     nn.ReLU())
        self.smooth3 = nn.Sequential(nn.ReLU(),
                                     nn.Conv2d(channels[2], channels[2], kernel_size=1),
                                     nn.ReLU())
        self.smooth2 = nn.Sequential(nn.ReLU(),
                                     nn.Conv2d(channels[1], channels[1], kernel_size=1),
                                     nn.ReLU())
        self.smooth1 = nn.Sequential(nn.ReLU(),
                                     nn.Conv2d(channels[0], channels[0], kernel_size=1),
                                     nn.ReLU())

        #                                                                       up + skip, out
        self.up1 = RDBUp(channels[4], channels[3], channels[3])  # 1024 + 1024, 1024
        self.up2 = RDBUp(channels[3], channels[2], channels[2])  # 1024 + 512, 512
        self.up3 = RDBUp(channels[2], channels[1], channels[1])  # 512 + 256, 256
        self.up4 = RDBUp(channels[1], channels[0], channels[0])  # 256 + 64, 64
        self.last_up = UpPad()
        self.conv_3x3 = nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1)
        self.pad_to = PadToX(32)

    def forward(self, x):
        # pad to multiples of 32
        diffX, diffY, x, = self.pad_to(x)

        orig = x
        # need 3 channels for the pretrained resnet
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
        # need the orig just for the size
        x = self.last_up(x, orig)
        x = self.conv_3x3(x)

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


class ConvUp(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(ConvUp, self).__init__()
        self.up = UpPad()
        self.conv1 = nn.Conv2d(in_channels1 + in_channels2, out_channels, kernel_size=3, padding=1,
                               padding_mode='reflect', bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1, x2)

        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        x = self.norm1(x)
        x = nn.LeakyReLU(inplace=True)(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = nn.LeakyReLU(inplace=True)(x)
        return x


class DualDenseNetUNetRDB(nn.Module):
    def __init__(self, features, global_features, v2=False):
        super(DualDenseNetUNetRDB, self).__init__()
        self.features = features
        self.global_features = global_features
        self.name = 'DualDenseUnetRDB'
        self.v2 = v2

        channels = [64, 256, 512, 1024, 1024]
        self.base_channel_size = channels[0]
        if not v2:
            self.smooth5 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(channels[4] * 2, channels[4], kernel_size=1),
                nn.ReLU(),
                RDB(n_channels=channels[4], nDenselayer=3, growthRate=channels[4] // 4)

            )
            self.smooth4 = nn.Sequential(nn.ReLU(),
                                         nn.Conv2d(channels[3], channels[3], kernel_size=1),
                                         nn.ReLU())
            self.smooth3 = nn.Sequential(nn.ReLU(),
                                         nn.Conv2d(channels[2], channels[2], kernel_size=1),
                                         nn.ReLU())
            self.smooth2 = nn.Sequential(nn.ReLU(),
                                         nn.Conv2d(channels[1], channels[1], kernel_size=1),
                                         nn.ReLU())
            self.smooth1 = nn.Sequential(nn.ReLU(),
                                         nn.Conv2d(channels[0], channels[0], kernel_size=1),
                                         nn.ReLU())
            self.up1 = RDBUp(channels[4], channels[3], channels[3])  # 1024 + 1024, 1024
            self.up2 = RDBUp(channels[3], channels[2], channels[2])  # 1024 + 512, 512
            self.up3 = RDBUp(channels[2], channels[1], channels[1])  # 512 + 256, 256
            self.up4 = RDBUp(channels[1], channels[0], channels[0])  # 256 + 64, 64
            self.last_up = UpPad()
            self.conv_3x3 = nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1)
        else:
            self.down5 = nn.Sequential(
                _Transition(channels[4] * 2, channels[4]),
                _DenseBlock(
                    num_layers=16,
                    num_input_features=channels[4],
                    bn_size=4,
                    growth_rate=32,
                    drop_rate=0,
                    memory_efficient=False
                )
            )
            self.up0 = ConvUp(channels[4] + 512, channels[4], channels[4])
            self.smooth4 = nn.Sequential(nn.BatchNorm2d(channels[3]),
                                         nn.LeakyReLU())
            self.smooth3 = nn.Sequential(nn.BatchNorm2d(channels[2]),
                                         nn.LeakyReLU())
            self.smooth2 = nn.Sequential(nn.BatchNorm2d(channels[1]),
                                         nn.LeakyReLU())
            self.smooth1 = nn.Sequential(nn.BatchNorm2d(channels[0]),
                                         nn.LeakyReLU())

            #                                                                       up + skip, out
            self.up1 = ConvUp(channels[4], channels[3], channels[3])  # 1024 + 1024, 1024
            self.up2 = ConvUp(channels[3], channels[2], channels[2])  # 1024 + 512, 512
            self.up3 = ConvUp(channels[2], channels[1], channels[1])  # 512 + 256, 256
            self.up4 = ConvUp(channels[1], channels[0], channels[0])  # 256 + 64, 64
        self.pad_to = PadToX(32)

    def forward(self, x):
        orig = x
        # need 3 channels for the pretrained resnet
        x = x.repeat(1, 3, 1, 1)
        gf = self.global_features(x)
        c1, c2, c3, c4, c5 = self.features(x)

        c5_cat = torch.cat([c5, gf[4]], 1)
        if self.v2:
            c6 = self.down5(c5_cat)
            c5 = self.up0(c6, c5)

        c1 = self.smooth1(c1)
        c2 = self.smooth2(c2)
        c3 = self.smooth3(c3)
        c4 = self.smooth4(c4)
        if not self.v2:
            c5 = self.smooth5(c5_cat)

        x = self.up1(c5, c4)
        x = self.up2(x, c3)
        x = self.up3(x, c2)
        x = self.up4(x, c1)
        if not self.v2:
            # need the orig just for the size
            x = self.last_up(x, orig)
            x = self.conv_3x3(x)

        return x

    def initialize(self):
        def init_layer(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.apply(init_layer)

        self.global_features.initialize()
        self.features.initialize()


@register
def DenseNet121UNetRDB():
    return DenseNetUNetRDB(features=DenseNet(32, (6, 12, 24, 16), 64, url=densenet.model_urls['densenet121']))


@register
def DualDenseNet121UNetRDB():
    return DualDenseNetUNetRDB(features=DenseNet(32, (6, 12, 24, 16), 64, url=None),
                               global_features=DenseNet(32, (6, 12, 24, 16), 64,
                                                        url=densenet.model_urls['densenet121']))


@register
def DualDenseNet121UNetRDBV2():
    return DualDenseNetUNetRDB(features=DenseNet(32, (6, 12, 24, 16), 64, url=None),
                               global_features=DenseNet(32, (6, 12, 24, 16), 64,
                                                        url=densenet.model_urls['densenet121']), v2=True)
