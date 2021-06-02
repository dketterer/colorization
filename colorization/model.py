import os

import torch
import torch.nn as nn

from colorization.backbones.utils import PadToX
from colorization.chkpt_utils import build_model_file_name
from colorization.backbones.heads import OutConv, TanHActivation
import colorization.backbones as backbones_mod
from colorization.backbones.smp_models import SMPModel


class Model(nn.Module):
    def __init__(self, backbone_name='ResUNet50_bc64', head_type='regression'):
        super(Model, self).__init__()

        self.name = f'ColorModel-{backbone_name}'
        self.backbone_name = backbone_name
        self.backbone = getattr(backbones_mod, self.backbone_name)()
        self.head_type = head_type

        def make_head(only_activation):
            if self.head_type.startswith('regression'):
                if only_activation:
                    return TanHActivation()
                return OutConv(self.backbone.base_channel_size, 2)

        self.head = make_head(only_activation=isinstance(self.backbone, SMPModel))
        self.up = None
        self.pad_to = PadToX(32)

    def forward(self, x):
        # pad to multiples of 32
        diffX, diffY, x, = self.pad_to(x)
        h, w = x.size()[2:]

        x = self.backbone(x)
        x = self.head(x)

        h_post, w_post = x.size()[2:]
        if h_post < h or w_post < w:
            if not self.up:
                scale_factor = h // h_post
                self.up = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=True)
            x = self.up(x)

        x = self.pad_to.remove_pad(x, diffX, diffY)
        return x

    def initialize(self):
        print('Init backbone')
        self.backbone.initialize()

        # Initialize head
        print('Init head')
        self.head.initialize()

    def __repr__(self):
        return '\n'.join([
            f'     model: {self.name}',
            f'  backbone: {self.backbone_name}',
            f'      head: {self.head_type}',
        ])

    def save(self, state, iteration):
        checkpoint = {
            'backbone_name': self.backbone_name,
            'head_type': self.head_type,
            'state_dict': self.state_dict()
        }

        for key in ('epoch', 'optimizer', 'scheduler', 'iteration', 'scaler', 'sampler'):
            if key in state:
                checkpoint[key] = state[key]

        # get real concrete save path:
        concrete_path = build_model_file_name(state['path'], iteration)
        assert not os.path.isfile(concrete_path)

        torch.save(checkpoint, concrete_path)

    @classmethod
    def load(cls, filename):
        if not os.path.isfile(filename):
            raise ValueError('No checkpoint {}'.format(filename))

        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
        # Recreate model from checkpoint instead of from individual backbones
        model = cls(backbone_name=checkpoint['backbone_name'], head_type=checkpoint['head_type'])
        model.load_state_dict(checkpoint['state_dict'])

        state = {}
        for key in ('epoch', 'optimizer', 'scheduler', 'iteration', 'scaler', 'sampler'):
            if key in checkpoint:
                state[key] = checkpoint[key]

        state['path'] = filename

        del checkpoint
        torch.cuda.empty_cache()

        return model, state
