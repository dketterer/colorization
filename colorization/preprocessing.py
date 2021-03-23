import cv2 as cv
import numpy as np
import math


def split_colors(image):
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    return lab[:, :, 0], lab[:, :, 1:]


def split_batch_ab_grey(batch):
    grey_batch = []
    ab_batch = []
    for image in batch:
        grey, ab = split_colors(image)
        grey_batch.append(grey)
        ab_batch.append(ab)
    grey_batch = np.array(grey_batch, dtype='float32')
    grey_batch = np.expand_dims(grey_batch, -1)
    grey_batch /= 255.
    # grey_batch -= 1.
    ab_batch = np.array(ab_batch, dtype='float32')
    ab_batch /= 128.
    ab_batch -= 1.
    return ab_batch, grey_batch


def resize(image, input_size):
    image = cv.resize(image, input_size, interpolation=cv.INTER_CUBIC)
    return image


def split_channels(lab):
    l = lab[:, :, 0].copy()
    ab = lab[:, :, 1:].copy()
    l = np.expand_dims(l, -1)
    return l, ab


def find_input_size(h, w):
    multiples_of = 16

    if h % multiples_of != 0:
        h -= (h % multiples_of)
    if w % multiples_of != 0:
        w -= (w % multiples_of)

    return h, w


def resize_keep_aspect_ratio(img, input_size):
    w, h, _ = img.shape
    if w > h:
        new_h = int(math.floor(h / w * input_size[1]))
        img = cv.resize(img, (new_h, input_size[1]), interpolation=cv.INTER_CUBIC)
    elif w < h:
        new_w = int(math.floor(w / h * input_size[0]))
        img = cv.resize(img, (input_size[0], new_w), interpolation=cv.INTER_CUBIC)
    else:
        img = cv.resize(img, input_size, interpolation=cv.INTER_CUBIC)
    return img


def zero_padd(img, input_size):
    w, h, _ = img.shape
    if w > h:
        left, right = 0, 0
        fill = (input_size[0] - h) / 2
        top = int(math.ceil(fill))
        bottom = int(math.floor(fill))
    elif w < h:
        top, bottom = 0, 0
        fill = (input_size[1] - w) / 2
        left = int(math.ceil(fill))
        right = int(math.floor(fill))
    else:
        top, bottom, left, right = 0, 0, 0, 0

    img = np.pad(img, ((left, right), (top, bottom), (0, 0)), 'constant', constant_values=-1)
    return img


def preprocessing(image):
    image = image.astype('float32')
    image /= 128.
    image -= 1.

    return image


def inference_prepro(image, input_size):
    # resize
    if input_size is not None:
        image = resize(image, input_size)
    # if 3ch -> split of L ch

    if image.shape[2] == 3:
        image, _ = split_channels(image)

    # divide by 128
    # subtract 1
    image = preprocessing(image)
    return image
