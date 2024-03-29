import copy
import json
import os
import re
import time
from typing import Callable, Optional, Sized, List

import cv2
import magic
import numpy as np
import torch

from torch.utils.data import Dataset, Sampler
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import ToTensor

from colorization.preprocessing import split_channels


def is_rgb(path):
    t = magic.from_file(path)
    if t.startswith('JPEG') and re.match(r'.*components 3.*', t):
        return True
    elif t.startswith('PNG') and re.match(r'.*8-bit/color RGB.*', t):
        return True
    return False


class ImagenetData(Dataset):
    def __init__(self,
                 folder: str,
                 rgb_json: str = None,
                 transform: Callable = None,
                 transform_l: Callable = None,
                 transform_ab: Callable = None,
                 training: bool = True,
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
        :param training: Indicate training or otherwise inference mode
        :type training: bool
        :param debug
        :type debug: bool
        """
        self.debug = debug
        self.imagenet_folder = folder
        self.transform = transform
        self.to_tensor_l = transform_l
        self.to_tensor_ab = transform_ab
        self.training = training

        image_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']

        imagenet_paths = []
        if rgb_json:
            with open(rgb_json, 'r') as f:
                rgb_file_names = json.load(f)
        tic = time.time()
        if rgb_json:
            rgb_file_names_set = set(rgb_file_names)

        for file in os.listdir(self.imagenet_folder):
            if os.path.isdir(os.path.join(self.imagenet_folder, file)):
                for file2 in os.listdir(os.path.join(self.imagenet_folder, file)):
                    if os.path.splitext(file2)[1] in image_extensions:
                        if not rgb_json or file2 in rgb_file_names_set:
                            imagenet_paths.append(os.path.join(self.imagenet_folder, file, file2))
            if os.path.splitext(file)[1] in image_extensions:
                if not rgb_json or file in rgb_file_names_set:
                    imagenet_paths.append(os.path.join(self.imagenet_folder, file))

        self.paths = imagenet_paths
        self.paths.sort()

        self.buffer_in_mem = False
        self.mem_dict = {}
        if len(self.paths) < 100:

            self.buffer_in_mem = True
            for img_path in self.paths:
                self.mem_dict[img_path] = np.fromfile(img_path, dtype=np.uint8)

        print(f'Loaded dataset in {time.time() - tic:.2f}s {"into memory" if self.buffer_in_mem else ""}')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = self.paths[item]
        if not self.buffer_in_mem:
            img_orig = cv2.imread(img_path, cv2.IMREAD_COLOR)

        else:
            img_orig = cv2.imdecode(self.mem_dict[img_path], cv2.IMREAD_COLOR)

        img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)['image']
            if not self.training:
                img_orig = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_orig = ToTensor()(img)

        if self.debug:
            plt.imshow(img)
            plt.show()

        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        grey_orig, ab_orig = split_channels(img)
        grey, ab = grey_orig, ab_orig
        if self.to_tensor_l:
            grey = self.to_tensor_l(grey)

        if self.to_tensor_ab:
            ab = self.to_tensor_ab(ab)

        if self.training:
            return grey, ab, img_orig

        return grey, ab, img_orig, grey_orig

    def collate_fn(self, batch):
        return default_collate(batch)


def extract_colorsegment_masks(segment: np.ndarray, n_segments) -> np.ndarray:
    """Extract the binary colorsegment masks from the image generated by the external tool.
    Each unique RGB (or BGR, does not matter) tuple is treated as a mask

    :param segment: np.ndarray (H, W, 3) uint8
    :type segment: np.ndarray
    :return: List[np.ndarray] dtype=bool, (H, W)
    :rtype:List[np.ndarray]
    """
    colors = np.unique(segment.reshape(-1, segment.shape[2]), axis=0)
    segment_masks = np.zeros((n_segments,) + segment.shape[:2], dtype=bool)
    for i in range(colors.shape[0]):
        segment_masks[i] = (segment == colors[i, :]).sum(2) == 3
    return segment_masks


class ImagenetColorSegmentData(Dataset):
    def __init__(self,
                 imagenet_folder: str,
                 colorsegment_folder: str,
                 rgb_json: str,
                 transform: Callable = None,
                 transform_l: Callable = None,
                 transform_ab: Callable = None,
                 training: bool = True,
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
        :param training: Indicate training or otherwise inference mode
        :type training: bool
        :param debug
        :type debug: bool
        """
        self.debug = debug
        self.imagenet_folder = imagenet_folder
        self.colorsegment_folder = colorsegment_folder
        self.transform = transform
        self.to_tensor_l = transform_l
        self.to_tensor_ab = transform_ab
        self.training = training

        image_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']

        imagenet_paths = []

        with open(rgb_json, 'r') as f:
            rgb_file_names = json.load(f)
        tic = time.time()

        file_names = copy.copy(rgb_file_names)
        rgb_file_names_set = set(rgb_file_names)

        for file in os.listdir(self.imagenet_folder):
            if os.path.isdir(os.path.join(self.imagenet_folder, file)):
                for file2 in os.listdir(os.path.join(self.imagenet_folder, file)):
                    if os.path.splitext(file2)[1] in image_extensions:
                        if file2 in rgb_file_names_set:
                            imagenet_paths.append(os.path.join(self.imagenet_folder, file, file2))
            if os.path.splitext(file)[1] in image_extensions:
                if file in rgb_file_names_set:
                    imagenet_paths.append(os.path.join(self.imagenet_folder, file))

        segment_paths = [os.path.join(self.colorsegment_folder, os.path.splitext(file)[0] + '.png') for file in
                         file_names]
        self.paths = list(zip(imagenet_paths, segment_paths))
        self.paths.sort(key=lambda pair: pair[0])

        self.buffer_in_mem = False
        self.mem_dict = {}
        if len(self.paths) < 100:

            self.buffer_in_mem = True
            for img_path, segment_path in self.paths:
                self.mem_dict[img_path] = np.fromfile(img_path, dtype=np.uint8)
                if os.path.isfile(segment_path):
                    self.mem_dict[segment_path] = np.fromfile(segment_path, dtype=np.uint8)
                else:
                    self.mem_dict[segment_path] = None
            self.paths = [path_tup for path_tup in self.paths if
                          self.mem_dict[path_tup[0]] is not None and self.mem_dict[path_tup[1]] is not None]
        print(f'Loaded dataset in {time.time() - tic:.2f}s {"into memory" if self.buffer_in_mem else ""}')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path, segment_path = self.paths[item]
        if not self.buffer_in_mem:
            img_orig = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if os.path.isfile(segment_path):
                segment = cv2.imread(segment_path, cv2.IMREAD_COLOR)[:, :, 0]
            else:
                raise
        else:
            img_orig = cv2.imdecode(self.mem_dict[img_path], cv2.IMREAD_COLOR)
            if self.mem_dict[segment_path] is not None:
                segment = cv2.imdecode(self.mem_dict[segment_path], cv2.IMREAD_COLOR)[:, :, 0]
            else:
                raise

        img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

        if self.transform:
            kwargs = {'mask': segment} if segment is not None else {}
            transformed = self.transform(image=img, **kwargs)
            img = transformed['image']
            if segment is not None:
                segment = transformed['mask']
                segment = segment.astype(np.int64)
            if not self.training:
                img_orig = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_orig = ToTensor()(img)

        if self.debug:
            plt.imshow(img)
            plt.show()

        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        grey_orig, ab_orig = split_channels(img)
        grey, ab = grey_orig, ab_orig
        if self.to_tensor_l:
            grey = self.to_tensor_l(grey)

        if self.to_tensor_ab:
            ab = self.to_tensor_ab(ab)

        if self.training:
            return grey, ab, segment, img_orig

        return grey, ab, img_orig, grey_orig


def get_trainloader(dataset, batch_size, sampler):
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       sampler=sampler,
                                       num_workers=8,
                                       pin_memory=True,
                                       prefetch_factor=2)


class SavableShuffleSampler(Sampler):
    def __init__(self, data_source: Optional[Sized], shuffle=True):
        super().__init__(data_source)
        self.data_source = data_source
        self.generator = torch.Generator()
        self.generator.manual_seed(1337)
        self.epoch = 0
        self.do_shuffle = shuffle

        self.new_seq()

    def new_seq(self):
        self.index = 0
        self.epoch += 1
        if self.do_shuffle:
            self.seq = torch.randperm(len(self.data_source), generator=self.generator)
        else:
            self.seq = torch.arange(len(self.data_source))

    def __iter__(self):
        if self.index >= len(self.seq):
            self.new_seq()

        while self.index < len(self.seq):
            self.index += 1
            yield self.seq[self.index - 1]

    def __len__(self):
        return len(self.seq)

    def state_dict(self):
        return {'seq': self.seq, 'index': self.index, 'generator': self.generator.get_state(),
                'do_shuffle': self.do_shuffle, 'epoch': self.epoch}

    def load_state_dict(self, state_dict):
        self.seq = state_dict['seq']
        self.index = state_dict['index']
        self.generator.set_state(state_dict['generator'])
        self.do_shuffle = state_dict['do_shuffle']
        self.epoch = state_dict.get('epoch', 0)
