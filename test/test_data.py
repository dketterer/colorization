import unittest

import cv2
import numpy as np

from colorization import data


class TestImagenetData(unittest.TestCase):
    def test_basics(self):
        transform = None
        dataloader = data.ImagenetData('test/resources/images', transform)
        ab, l = dataloader[0]
        # basics
        self.assertEqual(np.uint8, l.dtype)
        self.assertEqual(np.uint8, ab.dtype)
        self.assertEqual(np.ndarray, type(l))
        self.assertEqual(np.ndarray, type(ab))
        self.assertEqual(5, len(dataloader))

    def test_conversion(self):
        transform = None
        dataloader = data.ImagenetData('test/resources/images', transform)
        # targets l-pixel.png
        ab, l = dataloader[3]
        self.assertEqual(100, np.round(l[:, :, -1]))
        self.assertEqual(0, np.round(ab[:, :, -2]))
        self.assertEqual(0, np.round(ab[:, :, -1]))

    def test_color(self):
        bgr = cv2.imread('test/resources/images/ILSVRC2012_test_00000010.JPEG')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        back_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        try:
            np.testing.assert_allclose(back_bgr, bgr, atol=6, rtol=255)
        except Exception:
            self.fail()

if __name__ == '__main__':
    unittest.main()
