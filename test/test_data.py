import copy
import unittest

import cv2
import numpy as np
import torch
import albumentations as A

from colorization import data
from colorization.train import to_tensor_l, to_tensor_ab
from colorization.data import SavableShuffleSampler, get_trainloader


class TestImagenetData(unittest.TestCase):
    def test_basics(self):
        transform = None
        dataset = data.ImagenetData('test/resources/images', transform)
        l, ab = dataset[0]
        # basics
        self.assertEqual(np.uint8, l.dtype)
        self.assertEqual(np.uint8, ab.dtype)
        self.assertEqual(np.ndarray, type(l))
        self.assertEqual(np.ndarray, type(ab))
        self.assertEqual(7, len(dataset))

    def test_conversion(self):
        transform = None
        dataset = data.ImagenetData('test/resources/images', transform)
        dataset.training = True
        # targets images/a/black-pixel.png
        l, ab = dataset[4]
        self.assertEqual(0, l[:, :, -1])
        self.assertEqual(128, ab[:, :, -2])
        self.assertEqual(128, ab[:, :, -1])

    def test_color(self):
        bgr = cv2.imread('test/resources/images/a/ILSVRC2012_test_00000008.JPEG')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        back_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        try:
            np.testing.assert_allclose(back_bgr, bgr, atol=6, rtol=255)
        except Exception:
            self.fail()

    def test_white(self):
        transform = None
        dataset = data.ImagenetData('test/resources/images',
                                    transform,
                                    transform_l=to_tensor_l,
                                    transform_ab=to_tensor_ab)
        # targets white-pixel.png
        l, ab = dataset[6]
        self.assertEqual(1., l.numpy())
        try:
            np.testing.assert_allclose(ab.numpy().squeeze(), np.array([0., 0.]), atol=0.05, rtol=0.001)
        except Exception:
            self.fail()
        # targets black-pixel.png
        l, ab = dataset[3]
        self.assertEqual(-1., l.numpy())
        try:
            np.testing.assert_allclose(ab.numpy().squeeze(), np.array([0., 0.]), atol=0.05, rtol=0.001)
        except Exception:
            self.fail()

    def test_SaveableSampler(self):
        data = list(range(10))

        sampler = SavableShuffleSampler(data)
        next(iter(sampler))

        seq = copy.deepcopy(sampler.seq)

        state = sampler.state_dict()
        del sampler

        new_sampler = SavableShuffleSampler(data)
        new_sampler.load_state_dict(state)

        self.assertEqual(1, new_sampler.index)
        self.assertTrue(torch.all(torch.eq(seq, new_sampler.seq)))

        # consume the rest of the sampler
        iterator = iter(new_sampler)
        for _ in range(9):
            next(iterator)

        # trigger reshuffle
        next(iter(new_sampler))

        self.assertFalse(torch.all(torch.eq(seq, new_sampler.seq)))

        # test no shuffle mode
        no_shuffle_sampler = SavableShuffleSampler(data, shuffle=False)

        result = [idx for idx in no_shuffle_sampler]

        self.assertEqual(data, result)

    def test_trainloader(self):
        transform = A.Compose(
            [A.Resize(256, 256, always_apply=True)])
        dataset = data.ImagenetData('test/resources/images', transform)
        trainloader = get_trainloader(dataset, 7, SavableShuffleSampler(dataset, shuffle=False))
        it = iter(trainloader)
        d = it.__next__()
        inputs, ab = d
        self.assertEqual(7, len(inputs))
        self.assertEqual(7, len(ab))

    def test_imagenet_with_color_segments(self):
        transform = A.Compose(
            [A.Resize(256, 256, always_apply=True)])
        dataset = data.ImagenetColorSegmentData('test/resources/images', 'test/resources/color_segment', transform)
        l, ab, segment_masks = dataset[0]
        # basics
        self.assertEqual([], segment_masks)
        l, ab, segment_masks = dataset[1]
        self.assertTrue(isinstance(segment_masks[0], np.ndarray))
        self.assertEqual(50, len(segment_masks))

    def test_trainloader_with_color_segments(self):
        transform = A.Compose(
            [A.Resize(256, 256, always_apply=True)])
        dataset = data.ImagenetColorSegmentData('test/resources/images', 'test/resources/color_segment', transform)
        trainloader = get_trainloader(dataset, 7, SavableShuffleSampler(dataset, shuffle=False))
        it = iter(trainloader)
        d = it.__next__()
        inputs, labels = d
        ab, segment_masks = labels
        self.assertEqual(7, len(inputs))
        self.assertEqual(7, len(segment_masks))
        self.assertEqual(7, len(ab))
        self.assertEqual(50, len(segment_masks[1]))


if __name__ == '__main__':
    unittest.main()
