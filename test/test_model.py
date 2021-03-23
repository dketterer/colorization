import unittest

from colorization.model import Model


class TestModel(unittest.TestCase):
    def test_structure(self):
        model = Model(backbone='UNet_bc64_d4', head_type='regression')

        self.assertEqual('UNet', model.backbone.name)
        self.assertEqual(64, model.head.conv.in_channels)
        self.assertEqual(2, model.head.conv.out_channels)


if __name__ == '__main__':
    unittest.main()
