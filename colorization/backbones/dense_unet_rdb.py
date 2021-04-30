from torch import nn
from torch.cuda.amp import autocast
from torchvision.models import densenet
import torch.nn.functional as F

from colorization.backbones.densenet import DenseNet
from colorization.backbones.utils import register
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

    def forward(self, x):
        # pad to multiples of 32
        pad_to = 32
        w, h = x.size()[2:]
        w_padded = w + pad_to - w % pad_to if w % pad_to != 0 else w
        h_padded = h + pad_to - h % pad_to if h % pad_to != 0 else h
        diffY = w_padded - w
        diffX = h_padded - h
        if diffX > 0 or diffY > 0:
            # Hack for https://github.com/pytorch/pytorch/issues/13058
            with autocast(enabled=False):
                x = x.float()

                x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])

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

        if diffX > 0 or diffY > 0:
            # remove the pad
            x = x[:, :, diffY // 2:x.size()[2] - (diffY - diffY // 2), diffX // 2:x.size()[3] - (diffX - diffX // 2)]

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
def DenseNet121UNetRDB():
    return DenseNetUNetRDB(features=DenseNet(32, (6, 12, 24, 16), 64, url=densenet.model_urls['densenet121']))
