import os

import cv2
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import scipy.stats as st

from colorization.model import Model
from colorization.data import ImagenetData
from colorization.preprocessing import to_tensor_l, to_tensor_ab
from colorization.metrics import create_window


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


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def infer(model: Model,
          target_path: str,
          dataset: Dataset = None,
          image_path: str = '',
          batch_size: int = 8,
          img_limit: int = 50,
          debug: bool = False,
          transform=None,
          tensorboard: bool = False):
    if not os.path.exists(target_path):
        os.makedirs(target_path, exist_ok=True)
    assert bool(dataset) != bool(image_path), 'Specify only one: dataset or image_path'
    if not dataset and image_path:
        dataset = ImagenetData(image_path, transform=transform, transform_l=to_tensor_l, transform_ab=to_tensor_ab,
                               training=False)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            prefetch_factor=batch_size * 2)

    results = []

    gaussian_filter = create_window(7, batch_size, gaussian_weights=True)
    if torch.cuda.is_available():
        gaussian_filter = gaussian_filter.cuda()

    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    pbar = tqdm(dataloader, leave=not tensorboard)
    img_index = 0
    for i, data in enumerate(pbar):
        grey, _, img_orig, grey_orig = data

        if torch.cuda.is_available():
            grey = grey.cuda()
        if debug:
            grey_smooth = F.conv2d(grey, gaussian_filter, padding=3, groups=batch_size)
        with torch.no_grad():
            with autocast():
                prediction = model(grey)
                prediction_smooth = model(grey_smooth)
        del grey
        if debug:
            del grey_smooth

        prediction = prediction.to('cpu:0').numpy()
        prediction = np.transpose(prediction, (0, 2, 3, 1))
        if debug:
            prediction_smooth = prediction_smooth.to('cpu:0').numpy()
            prediction_smooth = np.transpose(prediction_smooth, (0, 2, 3, 1))
        grey_orig = grey_orig.numpy()
        img_orig = img_orig.numpy()

        for batch_idx in range(prediction.shape[0]):
            if debug:
                img_result = stich_debug_image(grey_orig[batch_idx, ...], prediction[batch_idx, ...],
                                               prediction_smooth[batch_idx, ...], img_orig[batch_idx, ...])
            else:
                img_result = stich_image(grey_orig[batch_idx, ...], prediction[batch_idx, ...])

            if tensorboard:
                results.append(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))

            cv2.imwrite(os.path.join(target_path, f'prediction-{img_index:05d}.jpg'), img_result)
            img_index += 1
        if (i + 1) * batch_size >= img_limit:
            break
    if tensorboard:
        return results


def stich_debug_image(grey_orig, ab_pred, ab_smooth, img_orig) -> np.ndarray:
    h, w = grey_orig.shape[:2]
    result = np.empty((h, w * 4, 3), dtype='uint8')
    result[:, :w, :] = img_orig
    result[:, w:w * 2, :] = cv2.cvtColor(grey_orig, cv2.COLOR_GRAY2BGR)

    predicted = stich_image(grey_orig, ab_pred)
    predicted_smooth = stich_image(grey_orig, ab_smooth)

    result[:, 2 * w:3 * w, :] = predicted
    result[:, 3 * w:, :] = predicted_smooth

    return result
