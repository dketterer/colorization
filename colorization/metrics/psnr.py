import torch


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [-1,1]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        img1 /= 2.
        img1 += 0.5
        img1 *= 255.
        img2 /= 2.
        img2 += 0.5
        img2 *= 255.
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))