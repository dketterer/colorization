import unittest

import torch

from colorization.backbones.unet_dual_encoder import UNetDualEncoderVGG16


class TestUNet(unittest.TestCase):
    def test_model_structure(self):
        model = UNetDualEncoderVGG16()
        model.initialize()

        inp = torch.rand(1, 1, 256, 256)
        out = model(inp)


if __name__ == '__main__':
    unittest.main()
