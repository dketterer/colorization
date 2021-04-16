import torch
from torch import nn


def psnr_func(img1, img2):
    img1_copy = img1.divide(2)
    img2_copy = img2.divide(2)
    img1_copy += 0.5
    img1_copy *= 255.
    img2_copy += 0.5
    img2_copy *= 255.
    mse = torch.mean((img1_copy - img2_copy) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse))


class PSNR(nn.Module):
    """Peak Signal to Noise Ratio
    img1 and img2 have range [-1,1]"""

    def __init__(self):
        super().__init__()
        self.name = "PSNR"

    def forward(self, img1, img2):
        return psnr_func(img1, img2)
