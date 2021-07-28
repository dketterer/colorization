from torch import nn

from colorization.backbones.utils import register
import segmentation_models_pytorch as smp


class SMPModel(nn.Module):
    def __init__(self, features):
        super(SMPModel, self).__init__()
        self.features = features
        self.name = features.name

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        return self.features(x)

    def initialize(self):
        def initialize_decoder(module):
            for m in module.modules():

                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        def initialize_head(module):
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight, nn.init.calculate_gain('tanh'), nonlinearity='tanh')
                    nn.init.zeros_(m.bias)

        initialize_decoder(self.features.decoder)
        initialize_head(self.features.segmentation_head)


@register
def InceptionResNetV2PSPNet():
    return SMPModel(smp.PSPNet(encoder_name='inceptionresnetv2', encoder_weights='imagenet', classes=2))


@register
def InceptionResNetV2UNet():
    return SMPModel(smp.Unet(encoder_name='inceptionresnetv2', encoder_weights='imagenet', classes=2))


@register
def InceptionResNetV2UNetAttention():
    return SMPModel(smp.Unet(encoder_name='inceptionresnetv2', encoder_weights='imagenet', classes=2,
                             decoder_attention_type='scse'))


@register
def Resnet50UNetAttention():
    return SMPModel(smp.Unet(encoder_name='resnet50', encoder_weights='imagenet', classes=2,
                             decoder_attention_type='scse'))


@register
def Resnet34UNetAttention():
    return SMPModel(smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', classes=2,
                             decoder_attention_type='scse'))
