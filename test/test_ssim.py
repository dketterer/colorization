import unittest

import numpy as np
import torch

from colorization.metrics import SSIM


class TestSSIM(unittest.TestCase):
    def test_same_images(self):
        ssim = SSIM()

        img1 = torch.zeros([1, 3, 30, 30])
        img2 = torch.zeros([1, 3, 30, 30])

        res = ssim(img1, img2)

        self.assertEqual(1.0, res)

    def test_unequal(self):
        ssim = SSIM()

        img1 = torch.ones([1, 3, 30, 30])
        img2 = torch.ones([1, 3, 30, 30]) * -1

        res = ssim(img1, img2).numpy()
        try:
            np.testing.assert_allclose(0.0, res, atol=0.0003, rtol=0.001)
        except:
            self.fail()

    def test_cuda(self):
        ssim = SSIM()
        if torch.cuda.is_available():
            img1 = torch.zeros([1, 3, 30, 30]).cuda()
            img2 = torch.zeros([1, 3, 30, 30]).cuda()

            res = ssim(img1, img2)

            self.assertEqual(1.0, res)


if __name__ == '__main__':
    unittest.main()