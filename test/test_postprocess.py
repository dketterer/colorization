import unittest

import torch
import torchvision.transforms.transforms

from colorization.cielab import rgb2lab
from colorization.postprocess import stack_input_predictions_to_rgb


class TestPostprocess(unittest.TestCase):
    def test_stack_input_predictions_to_rgb(self):
        orig = torch.zeros((16, 50, 50, 3)).float()
        # red image
        orig[..., 0] = 255

        transform = torchvision.transforms.transforms.Normalize((0.5,), (0.5,))

        lab_orig = rgb2lab(orig / 255)

        inputs = torch.unsqueeze(transform(lab_orig[..., 0] / 100), -1)
        outputs = (lab_orig[..., 1:] + 0.5) / 127.5

        rgb_reconstructed = stack_input_predictions_to_rgb(inputs.permute(0, 3, 1, 2), outputs.permute(0, 3, 1, 2))

        self.assertTrue(torch.all(torch.isclose(orig, rgb_reconstructed, rtol=1e-05, atol=4e-05)))


if __name__ == '__main__':
    unittest.main()
