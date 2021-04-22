import albumentations as A
from colorization.augmentation import LimitMaxSize


def get_transform(size):
    transform = A.Compose(
        [A.RandomResizedCrop(size, size, scale=(0.7, 1.0), ratio=(0.9, 1.1), always_apply=True),
         A.RandomRotate90(p=0.3),
         A.HorizontalFlip(p=0.3),
         A.VerticalFlip(p=0.1)
         ])
    return transform


def get_val_transform(size=None):
    compose = [LimitMaxSize()] if size else []
    transform = A.Compose(
        [*compose])
    return transform
