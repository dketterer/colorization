import unittest

import torch
from torchvision import models

from colorization.backbones.densenet import DenseNet


class TestDenseNet(unittest.TestCase):
    def test_forward(self):
        densenet = DenseNet(32, (6, 12, 24, 16), 64, url=models.densenet.model_urls['densenet121'])

        inp = torch.rand(1, 3, 512, 512)

        out = densenet(inp)
        print(len(out))
        for c in out:
            print(c.size())


if __name__ == '__main__':
    unittest.main()
