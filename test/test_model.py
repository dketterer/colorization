import unittest

from colorization.model import Model


class TestModel(unittest.TestCase):
    def test_structure(self):
        model = Model(backbone_name='ResUNet50_bc64', head_type='regression')

        self.assertEqual('ResUNet', model.backbone.name)
        self.assertEqual(64, model.head.conv.in_channels)
        self.assertEqual(2, model.head.conv.out_channels)


if __name__ == '__main__':
    unittest.main()
