import unittest

import cv2
import torch

from colorization.metrics import PSNR, PSNR_RGB


class TestPSNR(unittest.TestCase):
    def test_structure(self):
        psnr = PSNR()

        input = torch.rand(3, 2, 30, 30)
        target = torch.rand(3, 2, 30, 30)

        _psnr = psnr(input, target)

        self.assertGreater(70.0, _psnr)
        self.assertLess(0.0, _psnr)

    def test_structure_rgb(self):
        psnr = PSNR_RGB()

        input = torch.rand(3, 2, 30, 30)
        target = torch.rand(3, 2, 30, 30)
        l = torch.rand(3, 1, 30, 30)

        _psnr = psnr(input, target, l)

        self.assertGreater(70.0, _psnr)
        self.assertLess(0.0, _psnr)

    def test_psnr_rgb(self):
        img1 = cv2.imread('resources/images/a/ILSVRC2012_test_00000008.JPEG')
        lab = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB).astype(float)
        # normalize
        lab /= 255.
        lab -= 0.5
        lab /= .5

        lab = torch.from_numpy(lab).permute(2, 0, 1).unsqueeze(0).half().cuda()

        psnr = PSNR_RGB()
        ab = lab[:, 1:, ...]
        rand = torch.rand_like(ab)
        rand -= 0.5
        rand /= 0.5

        res = psnr(ab, rand, lab[:, 0, ...].unsqueeze(1))
        print(res)




if __name__ == '__main__':
    unittest.main()
