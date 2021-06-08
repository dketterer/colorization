import unittest

from colorization.backbones.unet import UNet_bc64_d4, UNet_bc64_d5, UNet_bc64_d3


class TestUNet(unittest.TestCase):
    def test_model_structure(self):
        model = UNet_bc64_d4()
        self.assertEqual(4, len(model.ups))
        self.assertEqual(4, len(model.downs))
        self.assertEqual(64, list(model.downs[0].maxpool_conv.modules())[2].in_channels)
        self.assertEqual(128, list(model.downs[0].maxpool_conv.modules())[2].out_channels)

        self.assertEqual(128, list(model.downs[1].maxpool_conv.modules())[2].in_channels)
        self.assertEqual(256, list(model.downs[1].maxpool_conv.modules())[2].out_channels)

        self.assertEqual(256, list(model.downs[2].maxpool_conv.modules())[2].in_channels)
        self.assertEqual(512, list(model.downs[2].maxpool_conv.modules())[2].out_channels)

        self.assertEqual(512, list(model.downs[3].maxpool_conv.modules())[2].in_channels)
        self.assertEqual(512, list(model.downs[3].maxpool_conv.modules())[2].out_channels)

        self.assertEqual(1024, model.ups[0].conv.in_channels)
        self.assertEqual(256, model.ups[0].conv.out_channels)

        self.assertEqual(512, model.ups[1].conv.in_channels)
        self.assertEqual(128, model.ups[1].conv.out_channels)

        self.assertEqual(256, model.ups[2].conv.in_channels)
        self.assertEqual(64, model.ups[2].conv.out_channels)

        self.assertEqual(128, model.ups[3].conv.in_channels)
        self.assertEqual(64, model.ups[3].conv.out_channels)

        model = UNet_bc64_d5()

        self.assertEqual(1024, list(model.downs[4].maxpool_conv.modules())[2].in_channels)
        self.assertEqual(1024, list(model.downs[4].maxpool_conv.modules())[2].out_channels)

        model = UNet_bc64_d3()

        self.assertEqual(256, list(model.downs[2].maxpool_conv.modules())[2].in_channels)
        self.assertEqual(256, list(model.downs[2].maxpool_conv.modules())[2].out_channels)


if __name__ == '__main__':
    unittest.main()
