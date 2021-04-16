import cv2
from albumentations import py3round, resize
from albumentations.augmentations import functional as F
from albumentations.core.transforms_interface import DualTransform


class LimitMaxSize(DualTransform):
    """Rescale an image so that maximum side is not longer than max_size, keeping the aspect ratio of the initial image.
    Args:
        max_size (int): maximum size of the image after the transformation.
        interpolation (OpenCV flag): interpolation method. Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """

    def __init__(self, max_size=1024, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1):
        super(LimitMaxSize, self).__init__(always_apply, p)
        self.interpolation = interpolation
        self.max_size = max_size

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        height, width = img.shape[:2]

        if self.max_size < float(max(width, height)):

            scale = self.max_size / float(max(width, height))

            if scale != 1.0:
                new_height, new_width = tuple(py3round(dim * scale) for dim in (height, width))
                img = resize(img, height=new_height, width=new_width, interpolation=interpolation)
        return img

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        height = params["rows"]
        width = params["cols"]

        scale = self.max_size / max([height, width])
        return F.keypoint_scale(keypoint, scale, scale)

    def get_transform_init_args_names(self):
        return ("max_size", "interpolation")
