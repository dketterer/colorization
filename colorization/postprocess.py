import cv2
import numpy as np
import torch

from colorization.cielab import lab2rgb


def stich_image(grey_orig, ab_pred) -> np.ndarray:
    h, w = grey_orig.shape[:2]
    fused = np.empty((h, w, 3), dtype='uint8')
    fused[:, :, 0] = np.squeeze(grey_orig)
    ab_pred /= 2.
    ab_pred += 0.5
    ab_pred *= 255.
    fused[:, :, 1:] = np.round(ab_pred).astype('uint8')

    predicted = cv2.cvtColor(fused, cv2.COLOR_LAB2BGR)

    return predicted


def stack_input_predictions_to_rgb(inputs, predictions):
    """

    :param inputs: (B, 1, H, W) z-normalized in [-1, 1] with mean=0.5 and std=0.5
    :type inputs: torch.Tensor
    :param predictions: (B, 2, H, W) in interval [-1, 1]
    :type predictions: torch.Tensor

    :returns torch.tensor: shape (B, H, W, 3) with RGB values in [0, 255]
    """
    inp_type = inputs.dtype
    unnormalized_inputs = torch.multiply(inputs, 0.5) + 0.5  # [interval now [0, 1]
    l = unnormalized_inputs * 100
    ab = predictions * 127.5 - 0.5

    # l in 0..100
    # ab in -128..127
    lab = torch.cat([l, ab], dim=1).permute(0, 2, 3, 1)  # (B, H, W, C)

    rgb = lab2rgb(lab.float()) * 255
    rgb = rgb.type(inp_type)
    return rgb
