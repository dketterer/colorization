import unittest

import torch

from colorization.backbones import Resnet50PixShuffle


class TestResnetPixShuffle(unittest.TestCase):
    def test_structure(self):
        model = Resnet50PixShuffle()

        input = torch.randn([1, 1, 512, 512])

        y_hat = model(input)

        self.assertEqual([1, 64, 512, 512], list(y_hat.size()))


if __name__ == '__main__':
    unittest.main()
