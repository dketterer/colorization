import unittest

import cv2
import numpy as np
import torch

from cielab import lab2rgb
from skimage.color import rgb2lab


class TestCIELAB(unittest.TestCase):
    def test_lab2rgb_cv2(self):
        img1 = cv2.imread('resources/images/a/ILSVRC2012_test_00000008.JPEG')
        rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        sk_lab = rgb2lab(rgb1)

        lab = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
        # l in 0..100
        # ab in -128..127
        lab = torch.from_numpy(np.concatenate([np.expand_dims(lab[..., 0], -1) * 100. / 255., lab[..., 1:] - 128.], 2)).float()#.divide(255)
        rgb = lab2rgb(lab, illuminant='D65')
        rgb *= 255.
        try:
            np.testing.assert_allclose(rgb, rgb1, atol=10.0, rtol=10.0)
        except:
            self.fail()

    def test_lab2rgb(self):
        img1 = cv2.imread('resources/images/a/ILSVRC2012_test_00000008.JPEG')
        rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        sk_lab = rgb2lab(rgb1)
        lab = torch.from_numpy(sk_lab).float()
        rgb = lab2rgb(lab, illuminant='D65')
        rgb = rgb.multiply_(255).round().byte()
        self.assertTrue(torch.all(torch.eq(rgb, torch.from_numpy(rgb1))))


if __name__ == '__main__':
    unittest.main()
