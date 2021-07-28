import unittest

import torch

from colorization.metrics.fid import FrechetInceptionDistance


class MyFrechetInceptionDistance(unittest.TestCase):
    def test_fid(self):
        inp = torch.zeros(16, 300, 300, 3)

        fid = FrechetInceptionDistance(16)
        fid.calc_activations1(inp)
        fid.calc_activations2(inp)
        result = fid.calculate_frechet_distance()
        self.assertTrue(torch.allclose(result, torch.tensor(0.0), atol=0.031))





if __name__ == '__main__':
    unittest.main()
