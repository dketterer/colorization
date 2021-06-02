import os
from random import shuffle

import cv2
import numpy as np
from cuda_slic import slic
from multiprocessing.dummy import Pool

from tqdm.contrib.concurrent import process_map

N_SEGMENTS = 100
COMPACTNESS = 5
MAX_ITER = 10


def cuda_slic(img_lab, n_segments, compactness, max_iter=3):
    labels = slic(img_lab, n_segments=n_segments, compactness=compactness, max_iter=max_iter, convert2lab=False,
                  enforce_connectivity=False)

    return labels


def worker(args):
    img_path, target_folder = args
    base = os.path.basename(img_path)
    if os.path.isfile(os.path.join(target_folder, base)):
        return
    img = cv2.imread(img_path)
    img = cv2.GaussianBlur(img, (7, 7), 1)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    labels = cuda_slic(img_lab, n_segments=N_SEGMENTS, compactness=COMPACTNESS, max_iter=MAX_ITER)
    cv2.imwrite(os.path.join(target_folder, f'{os.path.splitext(base)[0]}.png'), labels)


def main():
    parent_folder = '/mnt/data/datasets/color-overfit/n01440764'
    target_folder = '/mnt/data/datasets/Imagenet-SLIC-SEG100-COMP5-ITER10/color-overfit'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder, exist_ok=True)

    color_images = []
    for file in os.listdir(parent_folder):
        color_images.append((os.path.join(parent_folder, file), target_folder))

    shuffle(color_images)
    r = process_map(worker, color_images, max_workers=6, chunksize=50)


if __name__ == '__main__':
    main()
