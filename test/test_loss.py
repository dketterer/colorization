import unittest

import numpy as np
import torch

from colorization.loss import L1Loss, L2Loss


class TestLoss(unittest.TestCase):
    def test_l1_weighted(self):
        input = torch.ones(3, 2, 20, 30)
        target = torch.zeros(3, 2, 20, 30)
        target[0, 0, 0, 0] = 50
        l1 = L1Loss(weighted=True)
        l = l1(input, target, None)
        try:
            np.testing.assert_allclose(20*30*7.193e-9, l[0].item(), atol=0.003, rtol=0.001)
        except:
            self.fail()

    def test_l1_weighted_target_unchanged(self):
        input = torch.rand(3, 2, 20, 30)
        target = torch.rand(3, 2, 20, 30)
        orig_target = target.clone()
        orig_input = input.clone()
        l1 = L1Loss(weighted=True)
        l = l1(input, target, None)
        self.assertTrue(torch.all(torch.eq(target, orig_target)))
        self.assertTrue(torch.all(torch.eq(input, orig_input)))

    def test_l2(self):
        input = torch.ones(3, 2, 20, 30)
        target = torch.zeros(3, 2, 20, 30)
        l2 = L2Loss(weighted=False)
        l = l2(input, target, None)
        try:
            np.testing.assert_allclose(1., l[0].item(), atol=0.003, rtol=0.001)
        except:
            self.fail()



if __name__ == '__main__':
    unittest.main()
