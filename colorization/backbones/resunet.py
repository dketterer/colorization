from torch import nn

from colorization.backbones.unetparts import Up, Down
from colorization.backbones.utils import register
from colorization.backbones.resnet import ResNet
from torchvision.models import resnet as vrn


class ResUNet(nn.Module):
    def __init__(self, features, bilinear: bool = True):
        super(ResUNet, self).__init__()
        self.features = features

        is_light = self.features.bottleneck == vrn.BasicBlock
        channels = [64, 64, 128, 256, 512] if is_light else [64, 256, 512, 1024, 2048]
        factor = 2 if bilinear else 1

        self.top = nn.Sequential(
            nn.Conv2d(1, channels[0], 1),
            nn.ReLU(inplace=True),
        )

        self.bottom = nn.Sequential(
            nn.Conv2d(channels[4], channels[4], 1),
            nn.ReLU(inplace=True),
        )
        #                                                                            up + skip, out
        self.up1 = Up(channels[4] + channels[3], channels[4] // factor, bilinear)  # 2048 + 1024, 1024
        self.up2 = Up(channels[3] + channels[2], channels[3] // factor, bilinear)  # 1024 + 512, 512
        self.up3 = Up(channels[2] + channels[1], channels[2] // factor, bilinear)  # 512 + 256, 256
        self.up4 = Up(channels[1] + channels[0], channels[1] // factor, bilinear)  # 256 + 64, 128
        self.up5 = Up(channels[1] // factor + channels[0], channels[0], bilinear)  # 256 + 64, 64

    def forward(self, x):
        # need 3 channels for the pretrained resnet
        top = self.top(x)
        x = x.repeat(1, 3, 1, 1)
        c1, c2, c3, c4, c5 = self.features(x)

        bottom = self.bottom(c5)
        x = self.up1(bottom, c4)
        x = self.up2(x, c3)
        x = self.up3(x, c2)
        x = self.up4(x, c1)
        x = self.up5(x, top)

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
def Resnet50_UNet():
    return ResUNet(features=ResNet(layers=[3, 4, 6, 3], bottleneck=vrn.Bottleneck, url=vrn.model_urls['resnet50']))


@register
def Resnext50_UNet():
    return ResUNet(features=ResNet(layers=[3, 4, 6, 3], bottleneck=vrn.Bottleneck, groups=32, width_per_group=4,
                                   url=vrn.model_urls['resnext50_32x4d']))
