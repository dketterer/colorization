import unittest
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


if __name__ == '__main__':
    unittest.main()
