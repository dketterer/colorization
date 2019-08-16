import numpy as np
import cv2.cv2 as cv2


def postprocess(pred_ab, l):
    result = np.stack([l[..., 0], pred_ab[..., 0], pred_ab[..., 1]], axis=-1)

    result += 1.
    result *= 128.
    result = result.astype('uint8')
    result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)

    return result
