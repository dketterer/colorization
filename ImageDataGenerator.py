import os
import random
from random import shuffle
from multiprocessing import Queue, Process

import cv2.cv2 as cv2
import numpy as np

from preprocessing import resize, split_channels, preprocessing, find_input_size, zero_padd, resize_keep_aspect_ratio


def worker(q_1, q_2, input_size):
    while True:
        images = q_1.get()

        grey_batch, ab_batch = [], []
        for img_path in images:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if len(img.shape) == 2:
                continue

            img = resize_keep_aspect_ratio(img, input_size)

            grey, ab = split_channels(img)

            grey = preprocessing(grey)
            grey = zero_padd(grey, input_size)

            ab = preprocessing(ab)
            ab = zero_padd(ab, input_size)

            grey_batch.append(grey)
            ab_batch.append(ab)
        if len(grey_batch) != len(images):
            continue
        grey_batch = np.array(grey_batch)
        ab_batch = np.array(ab_batch)
        q_2.put((grey_batch, ab_batch))


def coordinator(q_1: Queue, paths, batch_size):
    random.seed(1337)
    n = len(paths)
    i = 0
    while True:
        batch = []
        for _ in range(batch_size):
            if i == 0:
                shuffle(paths)
            batch.append(paths[i])
            i += 1
            if i == n:
                i = 0

        q_1.put(batch)


class ImageDataGenerator(object):
    def __init__(self, folder: str, image_size: (int, int), batch_size, workers: int):
        self.folder = folder
        self.image_size = image_size
        self.paths = []

        image_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG']

        for file in os.listdir(self.folder):
            if os.path.isdir(os.path.join(self.folder, file)):
                for file2 in os.listdir(os.path.join(self.folder, file)):
                    if os.path.splitext(file2)[1] in image_extensions:
                        self.paths.append(os.path.join(self.folder, file, file2))
            if os.path.splitext(file)[1] in image_extensions:
                self.paths.append(os.path.join(self.folder, file))

        self.q_1 = Queue(workers)
        self.q_2 = Queue(workers)

        self.processes = [Process(target=coordinator, args=(self.q_1, self.paths, batch_size))]

        for _ in range(workers):
            self.processes.append(Process(target=worker, args=(self.q_1, self.q_2, image_size)))

        for p in self.processes:
            p.start()

    @property
    def size(self):
        return len(self.paths)

    def generate(self, batch_size):
        while True:
            tup = self.q_2.get()
            grey_batch, ab_batch = tup
            yield (grey_batch, ab_batch)

    def __del__(self):
        for p in self.processes:
            p.kill()
