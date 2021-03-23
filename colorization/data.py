import os
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from colorization.preprocessing import split_channels


class ImagenetData(Dataset):
    def __init__(self,
                 folder: str,
                 transform: Callable = None,
                 transform_l: Callable = None,
                 transform_ab: Callable = None,
                 debug: bool = False):
        """

        :param folder: Path to the Imagenet folder with the images (test, train), can have sub-folders with images
        :type folder: str
        :param transform: Albumentations transform function
        :type transform: Callable
        :param transform_l: PyTorch transform applied to the l channel(eg Convert to Tensor)
        :type transform_l: Callable
        :param transform_ab: PyTorch transform applied to the ab channel (eg Convert to Tensor)
        :type transform_ab: Callable
        :param debug
        :type debug: bool
        """
        self.debug = debug
        self.folder = folder
        self.paths = []
        self.transform = transform
        self.to_tensor_l = transform_l
        self.to_tensor_ab = transform_ab

        image_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']

        for file in os.listdir(self.folder):
            if os.path.isdir(os.path.join(self.folder, file)):
                for file2 in os.listdir(os.path.join(self.folder, file)):
                    if os.path.splitext(file2)[1] in image_extensions:
                        self.paths.append(os.path.join(self.folder, file, file2))
            if os.path.splitext(file)[1] in image_extensions:
                self.paths.append(os.path.join(self.folder, file))

        self.paths.sort()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = self.paths[item]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)['image']

        if self.debug:
            plt.imshow(img)
            plt.show()

        img = img.astype('float32')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img = img.astype('uint8')

        grey, ab = split_channels(img)

        if self.to_tensor_l:
            grey = self.to_tensor_l(grey)

        if self.to_tensor_ab:
            ab = self.to_tensor_ab(ab)

        return grey, ab

    def collate_fn(self, batch):
        greys, abs = zip(*batch)
        abs_stacked = torch.stack(abs)
        greys_stacked = torch.stack(greys)

        return greys_stacked, abs_stacked


def get_trainloader(dataset, batch_size):
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       pin_memory=True,
                                       prefetch_factor=batch_size)
