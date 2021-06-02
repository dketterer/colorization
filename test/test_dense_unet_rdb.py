import unittest

import torch

from colorization.backbones import DenseNet121UNetRDB, DualDenseNet121UNetRDB, DualDenseNet121UNetRDBV2
from colorization.model import Model


class TestDenseUNetRDB(unittest.TestCase):
    def test_padding(self):
        model = DenseNet121UNetRDB()

        input = torch.randn([5, 1, 375, 511])

        y_hat = model(input)

        self.assertEqual([5, 64, 375, 511], list(y_hat.size()))

        input = torch.randn([5, 1, 512, 511])

        y_hat = model(input)

        self.assertEqual([5, 64, 512, 511], list(y_hat.size()))

    def test_dual(self):
        model = DualDenseNet121UNetRDB()

        input = torch.randn([5, 1, 375, 511])

        y_hat = model(input)

        self.assertEqual([5, 64, 375, 511], list(y_hat.size()))

        input = torch.randn([5, 1, 512, 511])

        y_hat = model(input)

        self.assertEqual([5, 64, 512, 511], list(y_hat.size()))

    def test_dual_v2(self):
        model = Model(backbone_name='DualDenseNet121UNetRDBV2')

        input = torch.randn([5, 1, 375, 511])

        y_hat = model(input)

        self.assertEqual([5, 2, 375, 511], list(y_hat.size()))

        input = torch.randn([5, 1, 512, 511])

        y_hat = model(input)

        self.assertEqual([5, 2, 512, 511], list(y_hat.size()))


if __name__ == '__main__':
    unittest.main()
