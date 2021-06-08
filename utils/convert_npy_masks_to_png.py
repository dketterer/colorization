import os

import cv2
import numpy as np
from tqdm import tqdm


def main():
    dir = '/mnt/data/datasets/Imagenet-SLIC-SEG100-COMP5-ITER10/color-overfit'
    for file in tqdm(os.listdir(dir)):
        if file.endswith('.npy'):
            labels = np.load(os.path.join(dir, file))
            cv2.imwrite(os.path.join(dir, os.path.splitext(file)[0] + '.png'), labels)


if __name__ == '__main__':
    main()
