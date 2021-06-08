from collections import OrderedDict

import torch
from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.models.vgg import model_urls, make_layers, cfgs

from colorization.backbones.vgg import VGG
from colorization.backbones.unetparts import DoubleConv, Down, Up
from colorization.backbones.utils import register, PadToX


class UNetDualEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, base_channel_size: int = 64, bilinear=True, depth=4,
                 second_encoder='vgg16'):
        super(UNetDualEncoder, self).__init__()
        self.name = 'UNet'
        self.n_channels = in_channels
        self.base_channel_size = base_channel_size
        self.bilinear = bilinear
        self.depth = depth
        self.second_encoder_name = second_encoder
        self.second_encoder = VGG(make_layers(cfgs['D'], False), init_weights=False)

        self.inc = DoubleConv(in_channels, base_channel_size)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.fuse = nn.Sequential(nn.Conv2d(512 + 1024, 1024, kernel_size=1), nn.ReLU(inplace=True),
                                  DoubleConv(1024, 1024))
        self.pad_to = PadToX(32)
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
        diffX, diffY, x, = self.pad_to(x)
        x_exp = x.repeat(1, 3, 1, 1)
        extra_features = self.second_encoder(x_exp)
        x = self.inc(x)
        intermediates = []
        for layer in self.downs:
            intermediates.append(x)
            x = layer(x)
        x = torch.cat([x, extra_features], 1)
        x = self.fuse(x)
        for layer, intermediate in zip(self.ups, intermediates[::-1]):
            x = layer(x, intermediate)
        x = self.pad_to.remove_pad(x, diffX, diffY)
        return x

    def initialize(self):

        def init_layer(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

        self.apply(init_layer)
        state_dict = load_state_dict_from_url(model_urls[self.second_encoder_name],
                                              progress=True)
        # https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3
        model_dict = self.second_encoder.state_dict()
        # 1. filter out unnecessary keys
        state_dict = OrderedDict({k: v for k, v in state_dict.items() if k in model_dict})
        # 2. overwrite entries in the existing state dict
        model_dict.update(state_dict)
        # 3. load the new state dict
        self.second_encoder.load_state_dict(state_dict)

        # Fix params in the second encoder
        for param in self.second_encoder.parameters():
            param.requires_grad = False


@register
def UNetVGG16DualEncoder():
    return UNetDualEncoder(1, 64, True, 5, 'vgg16')
