import unittest

import numpy as np
import torch
from torch import Tensor
from torch.cuda.amp import autocast

from colorization.loss import ColorConsistencyLoss, L1CCLoss, ColorConsistencyLoss


class TestColorConsistencyLoss(unittest.TestCase):
    def test_squared(self):
        ab = torch.arange(40).reshape(2, 2, 2, 5).float()
        segments = torch.unsqueeze(
            torch.stack([torch.tensor([[True, True, True, True, True], [False, False, False, False, False]]),
                         torch.tensor([[False, False, False, False, False], [True, True, True, True, True]])]),
            0).repeat(2, 1, 1, 1)

        loss = ColorConsistencyLoss('square')

        l = loss(ab, segments)
        self.assertEqual(2.0, l.numpy())

    def test_euclidean(self):
        ab = torch.tensor([[[[5, 3, 3],
                             [2, 1, 0],
                             [10, 0, 0]],
                            [[1, -1, -3],
                             [0, 0, 0],
                             [3, 0, 0]]]], dtype=torch.float, requires_grad=True)
        segments = torch.unsqueeze(torch.stack([torch.tensor([[True, True, True],
                                                              [True, True, False],
                                                              [True, False, False]]),
                                                torch.tensor([[False, False, False],
                                                              [False, False, True],
                                                              [False, True, True]])]), 0)

        loss = ColorConsistencyLoss('euclidean')

        l: Tensor = loss(ab, segments)
        try:
            np.testing.assert_almost_equal(1.9665, l.detach().numpy(), 3)
        except:
            self.fail()

        l.backward()

    def test_squared_2(self):
        ab = torch.tensor([[[[5, 3, 3],
                             [2, 1, 0],
                             [10, 0, 0]],
                            [[1, -1, -3],
                             [0, 0, 0],
                             [3, 0, 0]]]], dtype=torch.float, requires_grad=True)
        segments = torch.unsqueeze(torch.tensor([[0, 0, 0],
                                                 [0, 0, 1],
                                                 [0, 1, 1]]), 0)
        loss = ColorConsistencyLoss('square')

        l = loss(ab, segments)
        try:
            np.testing.assert_almost_equal(4.0, l.detach().numpy(), 3)
        except:
            self.fail()

        l.backward()

        self.assertTrue(torch.all(torch.isclose(torch.tensor([[[[0.1111, -0.1111, -0.1111],
                                                                [-0.2222, -0.3333, 0.0000],
                                                                [0.6667, 0.0000, 0.0000]],

                                                               [[0.1111, -0.1111, -0.3333],
                                                                [0.0000, 0.0000, 0.0000],
                                                                [0.3333, 0.0000, 0.0000]]]]), ab.grad, rtol=1e-4,
                                                atol=1e-6)))

    def test_linear(self):
        ab = torch.tensor([[[[5, 3, 3],
                             [2, 1, 0],
                             [10, 0, 0]],
                            [[1, -1, -3],
                             [0, 0, 0],
                             [3, 0, 0]]]], dtype=torch.float, requires_grad=True)
        segments = torch.unsqueeze(torch.stack([torch.tensor([[True, True, True],
                                                              [True, True, False],
                                                              [True, False, False]]),
                                                torch.tensor([[False, False, False],
                                                              [False, False, True],
                                                              [False, True, True]])]), 0)
        loss = ColorConsistencyLoss('linear')

        l = loss(ab, segments)
        l.backward()
        try:
            # result is 1.2222 for real l1_loss, but smooth_l1_loss behaves a little different
            np.testing.assert_almost_equal(0.944, l.detach().numpy(), 3)
        except:
            self.fail()

    def test_compound_on_cuda(self):
        ab = torch.tensor([[[[5, 3, 3],
                             [2, 1, 0],
                             [10, 0, 0]],
                            [[1, -1, -3],
                             [0, 0, 0],
                             [3, 0, 0]]]], dtype=torch.float, requires_grad=True, device='cuda:0')
        segments = torch.unsqueeze(torch.stack([torch.tensor([[True, True, True],
                                                              [True, True, False],
                                                              [True, False, False]]),
                                                torch.tensor([[False, False, False],
                                                              [False, False, True],
                                                              [False, True, True]])]), 0).long().cuda()
        loss = L1CCLoss(0.5, 'linear').cuda()
        l, dic = loss(ab, ab, segments)
        l.backward()
        try:
            # result for ccl is 0.944, l1 is 0
            np.testing.assert_almost_equal(0.472, l.item(), 3)
        except:
            self.fail()

    def test_filled(self):
        ab = torch.tensor([[[[5, 3, 3],
                             [2, 1, 0],
                             [10, 0, 0]],
                            [[1, -1, -3],
                             [0, 0, 0],
                             [3, 0, 0]]]], dtype=torch.float, requires_grad=True)
        segments = torch.unsqueeze(torch.stack([torch.tensor([[True, True, True],
                                                              [True, True, False],
                                                              [True, False, False]]),
                                                torch.tensor([[False, False, False],
                                                              [False, False, True],
                                                              [False, True, True]]),
                                                torch.tensor([[False, False, False],
                                                              [False, False, False],
                                                              [False, False, False]]),
                                                torch.tensor([[False, False, False],
                                                              [False, False, False],
                                                              [False, False, False]])
                                                ]), 0)
        loss = ColorConsistencyLoss('linear')

        l = loss(ab, segments)
        try:
            # result is 1.2222 for real l1_loss, but smooth_l1_loss behaves a little different
            np.testing.assert_almost_equal(0.944, l.detach().numpy(), 3)
        except:
            self.fail()

    def test_squared_masks_V2(self):
        ab = torch.tensor([[[[5, 3, 3],
                             [2, 1, 0],
                             [10, 0, 0]],
                            [[1, -1, -3],
                             [0, 0, 0],
                             [3, 0, 0]]]], dtype=torch.float, requires_grad=True)
        segments = torch.unsqueeze(torch.tensor([[0, 0, 0],
                                                 [0, 0, 1],
                                                 [0, 1, 1]]), 0)
        loss = ColorConsistencyLoss('square')

        l = loss(ab, segments)
        try:
            np.testing.assert_almost_equal(4.0, l.detach().numpy(), 3)
        except:
            self.fail()

        l.backward()

    def test_linear_masks_v2(self):
        ab = torch.tensor([[[[5, 3, 3],
                             [2, 1, 0],
                             [10, 0, 0]],
                            [[1, -1, -3],
                             [0, 0, 0],
                             [3, 0, 0]]]], dtype=torch.float, requires_grad=True)
        segments = torch.unsqueeze(torch.tensor([[0, 0, 0],
                                                 [0, 0, 1],
                                                 [0, 1, 1]]), 0)
        loss = ColorConsistencyLoss('linear')

        l = loss(ab, segments)
        try:
            # result is 1.2222 for real l1_loss, but smooth_l1_loss behaves a little different
            np.testing.assert_almost_equal(0.944, l.detach().numpy(), 3)
        except:
            self.fail()

    def test_linear_masks_v2_batch(self):
        ab = torch.ones(3, 2, 30, 30)
        segments = torch.cat([torch.zeros(2, 30, 30), torch.arange(30 * 30).reshape(1, 30, 30)], 0).long()
        loss = ColorConsistencyLoss('linear')

        l = loss(ab, segments)


if __name__ == '__main__':
    unittest.main()
