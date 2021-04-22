from torchvision.models import resnet as vrn
import torch.utils.model_zoo as model_zoo


class ResNet(vrn.ResNet):
    'Deep Residual Network - https://arxiv.org/abs/1512.03385'

    def __init__(self, layers=[3, 4, 6, 3], bottleneck=vrn.Bottleneck, groups=1, width_per_group=64,
                 url=None):
        self.stride = 128
        self.bottleneck = bottleneck
        self.url = url

        kwargs = {'block': bottleneck, 'layers': layers, 'groups': groups,
                  'width_per_group': width_per_group}
        super().__init__(**kwargs)

    def initialize(self):
        if self.url:
            self.load_state_dict(model_zoo.load_url(self.url))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        outputs = [x]
        x = self.maxpool(x)
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            x = layer(x)
            outputs.append(x)

        return outputs
