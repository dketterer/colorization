from torch import nn

from colorization.backbones.utils import register
from .unetparts import DoubleConv, Down, Up


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

    def forward(self, x):
        x = self.inc(x)
        intermediates = []
        for layer in self.downs:
            intermediates.append(x)
            x = layer(x)
        for layer, intermediate in zip(self.ups, intermediates[::-1]):
            x = layer(x, intermediate)
        return x


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