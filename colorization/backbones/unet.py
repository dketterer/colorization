from torch import nn

from colorization.backbones.utils import register, PadToX
from colorization.backbones.unetparts import DoubleConv, Down, Up


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, base_channel_size: int = 64, bilinear=True, depth=4):
        super(UNet, self).__init__()
        self.name = 'UNet'
        self.n_channels = in_channels
        self.base_channel_size = base_channel_size
        self.bilinear = bilinear
        self.depth = depth

        self.inc = DoubleConv(in_channels, base_channel_size)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        down_channel = base_channel_size
        factor = 2 if bilinear else 1
        # go down:
        # 64 -> 128 -> 256 -> 512 -> 1024
        for i in range(1, self.depth):
            self.downs.append(Down(down_channel, down_channel * 2))
            down_channel *= 2
        self.downs.append(Down(down_channel, down_channel * 2 // factor))
        for i in range(1, self.depth):
            self.ups.append(Up(down_channel * 2, down_channel // factor, bilinear))
            down_channel = down_channel // 2
        self.ups.append(Up(down_channel * 2, base_channel_size, bilinear))
        self.pad_to = PadToX(32)

    def forward(self, x):
        diffX, diffY, x, = self.pad_to(x)
        x = self.inc(x)
        intermediates = []
        for layer in self.downs:
            intermediates.append(x)
            x = layer(x)
        for layer, intermediate in zip(self.ups, intermediates[::-1]):
            x = layer(x, intermediate)
        x = self.pad_to.remove_pad(x, diffX, diffY)
        return x

    def initialize(self):

        def init_layer(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

        self.apply(init_layer)


@register
def UNet_bc16_d3():
    return UNet(in_channels=1, base_channel_size=16, bilinear=True, depth=3)


@register
def UNet_bc64_d3():
    return UNet(in_channels=1, base_channel_size=64, bilinear=True, depth=3)


@register
def UNet_bc64_d4():
    return UNet(in_channels=1, base_channel_size=64, bilinear=True, depth=4)


@register
def UNet_bc64_d5():
    return UNet(in_channels=1, base_channel_size=64, bilinear=True, depth=5)


@register
def UNet_bc32_d4():
    return UNet(in_channels=1, base_channel_size=32, bilinear=True, depth=4)
