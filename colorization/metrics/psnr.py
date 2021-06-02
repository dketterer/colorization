import torch
from torch import nn

from colorization.cielab import lab2rgb


def psnr_func(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse))


class PSNR(nn.Module):
    """Peak Signal to Noise Ratio
    img1 and img2 have range [-1,1]"""

    def __init__(self):
        super(PSNR, self).__init__()
        self.name = "PSNR"

    def forward(self, img1, img2):
        img1_copy = img1.divide(2)
        img2_copy = img2.divide(2)
        img1_copy += 0.5
        img1_copy *= 255.
        img2_copy += 0.5
        img2_copy *= 255.
        return psnr_func(img1_copy, img2_copy)


class PSNR_RGB(nn.Module):
    """Peak Signal to Noise Ratio
    img1 and img2 have range [-1,1]"""

    def __init__(self):
        super().__init__()
        self.name = "PSNR_RGB"

    def forward(self, ab1, ab2, l):
        # ab1: (B, C, H, W) [-1, 1]
        # ab2: (B, C, H, W) [-1, 1]
        # l: (B, 1, H, W) [0, 1]
        dtype = ab1.type()
        # unnormalize
        ab1 = ab1.multiply(0.5)
        ab2 = ab2.multiply(0.5)
        ab1 += 0.5
        #ab1 *= 255.
        ab2 += 0.5
        #ab2 *= 255.
        l = l.multiply(0.5)
        l = l.add_(0.5)
        #l = l.multiply_(255)
        l = l.permute(0, 2, 3, 1)
        lab1 = torch.cat([l, ab1.permute(0, 2, 3, 1)], 3)
        rgb1 = lab2rgb(lab1.float()).type(dtype)
        lab2 = torch.cat([l, ab2.permute(0, 2, 3, 1)], 3)
        rgb2 = lab2rgb(lab2.float()).type(dtype)
        return psnr_func(rgb1, rgb2)
