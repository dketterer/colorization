import re
from typing import List

import torch
from torch import Tensor, nn
from torch.hub import load_state_dict_from_url
from torch.utils import model_zoo
import torch.nn.functional as F
from torchvision import models
from torchvision.models.densenet import _load_state_dict


class DenseNet(models.DenseNet):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False, url=None):
        super().__init__(growth_rate=growth_rate, block_config=block_config,
                         num_init_features=num_init_features, bn_size=bn_size, drop_rate=drop_rate,
                         num_classes=num_classes, memory_efficient=memory_efficient)
        self.block_config = block_config
        self.url = url

    def initialize(self):
        if self.url:
            _load_state_dict(self, self.url, True)
        else:
            def init_layer(m):
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)

            self.apply(init_layer)

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        outputs = [x]
        x = self.features.pool0(x)
        for i, num_layers in enumerate(self.block_config):
            x = self.features._modules[f'denseblock{i + 1}'](x)
            if i != len(self.block_config) - 1:
                outputs.append(x)
                x = self.features._modules[f'transition{i + 1}'](x)
        x = self.features.norm5(x)
        outputs.append(x)

        return outputs
