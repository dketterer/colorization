import unittest

import torch

from colorization.backbones import Resnet50_UNetV2


class TestResUNet(unittest.TestCase):
    def test_structure(self):

        resunet = Resnet50_UNetV2()

        input = torch.randn([1, 1, 512, 512])

        y_hat = resunet(input)

        self.assertEqual([1, 64, 512, 512], list(y_hat.size()))


if __name__ == '__main__':
    unittest.main()
