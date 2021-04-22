from torch import nn

from colorization.backbones.unetparts import UpPad
from colorization.backbones.residual_dense_block import RDB, RDBUp
from colorization.backbones.utils import register
from colorization.backbones.resnet import ResNet
from torchvision.models import resnet as vrn


class ResnetUNetRDB(nn.Module):
    def __init__(self, features, bilinear: bool = True):
        super(ResnetUNetRDB, self).__init__()
        self.features = features
        self.name = 'ResnetUNetRDB'

        is_light = self.features.bottleneck == vrn.BasicBlock
        channels = [64, 64, 128, 256, 512] if is_light else [64, 256, 512, 1024, 2048]
        self.base_channel_size = channels[0]
        factor = 2 if bilinear else 1

        self.smooth5 = nn.Sequential(
            nn.Conv2d(channels[4], channels[4], kernel_size=1),
            RDB(n_channels=channels[4], nDenselayer=3, growthRate=channels[4] // 4)
        )
        self.smooth4 = nn.Conv2d(channels[3], channels[3], kernel_size=1)
        self.smooth3 = nn.Conv2d(channels[2], channels[2], kernel_size=1)
        self.smooth2 = nn.Conv2d(channels[1], channels[1], kernel_size=1)
        self.smooth1 = nn.Conv2d(channels[0], channels[0], kernel_size=1)

        #                                                                       up + skip, out
        self.up1 = RDBUp(channels[4], channels[3], channels[4] // factor)  # 2048 + 1024, 1024
        self.up2 = RDBUp(channels[3], channels[2], channels[3] // factor)  # 1024 + 512, 512
        self.up3 = RDBUp(channels[2], channels[1], channels[2] // factor)  # 512 + 256, 256
        self.up4 = RDBUp(channels[1], channels[0], channels[1] // factor)  # 256 + 64, 128
        self.last_up = UpPad()
        self.conv_3x3 = nn.Conv2d(channels[1] // factor, channels[0], kernel_size=3, padding=1)

    def forward(self, x):
        # need 3 channels for the pretrained resnet
        orig = x
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
def Resnet50UNetRDB():
    return ResnetUNetRDB(
        features=ResNet(layers=[3, 4, 6, 3], bottleneck=vrn.Bottleneck, url=vrn.model_urls['resnet50']))
