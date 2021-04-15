import unittest

import cv2
import numpy as np

from colorization import data
from colorization.train import to_tensor_l, to_tensor_ab


class TestImagenetData(unittest.TestCase):
    def test_basics(self):
        transform = None
        dataset = data.ImagenetData('test/resources/images', transform)
        l, ab = dataset[0]
        # basics
        self.assertEqual(np.uint8, l.dtype)
        self.assertEqual(np.uint8, ab.dtype)
        self.assertEqual(np.ndarray, type(l))
        self.assertEqual(np.ndarray, type(ab))
        self.assertEqual(6, len(dataset))

    def test_conversion(self):
        transform = None
        dataset = data.ImagenetData('test/resources/images', transform)
        dataset.training = True
        # targets images/a/black-pixel.png
        l, ab = dataset[3]
        self.assertEqual(0, l[:, :, -1])
        self.assertEqual(128, ab[:, :, -2])
        self.assertEqual(128, ab[:, :, -1])

    def test_color(self):
        bgr = cv2.imread('test/resources/images/a/ILSVRC2012_test_00000008.JPEG')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        back_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        try:
            np.testing.assert_allclose(back_bgr, bgr, atol=6, rtol=255)
        except Exception:
            self.fail()

    def test_white(self):
        transform = None
        dataset = data.ImagenetData('test/resources/images',
                                    transform,
                                    transform_l=to_tensor_l,
                                    transform_ab=to_tensor_ab)
        # targets white-pixel.png
        l, ab = dataset[5]
        self.assertEqual(1., l.numpy())
        try:
            np.testing.assert_allclose(ab.numpy().squeeze(), np.array([0., 0.]), atol=0.05, rtol=0.001)
        except Exception:
            self.fail()
        # targets black-pixel.png
        l, ab = dataset[3]
        self.assertEqual(-1., l.numpy())
        try:
            np.testing.assert_allclose(ab.numpy().squeeze(), np.array([0., 0.]), atol=0.05, rtol=0.001)
        except Exception:
            self.fail()


if __name__ == '__main__':
    unittest.main()
