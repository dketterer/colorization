import sys

import torch
from torch import nn
from torch.cuda.amp import autocast
import torch.nn.functional as F


class PadToX(nn.Module):
    def __init__(self, multiples_of=32):
        super(PadToX, self).__init__()
        self.multiples_of = multiples_of

    def forward(self, x: torch.Tensor) -> (float, float, torch.Tensor):
        pad_to = self.multiples_of
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
        return diffX, diffY, x

    def remove_pad(self, x, diffX, diffY):
        if diffX > 0 or diffY > 0:
            # remove the pad
            x = x[:, :, diffY // 2:x.size()[2] - (diffY - diffY // 2), diffX // 2:x.size()[3] - (diffX - diffX // 2)]
        return x


def register(f):
    all = sys.modules[f.__module__].__dict__.setdefault('__all__', [])
    if f.__name__ in all:
        raise RuntimeError('{} already exist!'.format(f.__name__))
    all.append(f.__name__)
    return f
