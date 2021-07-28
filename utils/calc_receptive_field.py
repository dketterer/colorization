import unittest

import numpy as np
import torch

from backbones import UNetVGG16DualEncoder
from colorization.backbones import Resnet50_UNetV2, Resnet50UNetRDB, Resnext50_UNet, Resnet50_UNet, Resnet101_UNetV2, \
    Resnet152_UNetV2, DenseNet121UNetRDB, InceptionResNetV2UNetAttention


class TestReceptiveField():
    def analyze(self, y_hat_1, y_hat_2, inp, name):
        print(name)
        print((np.abs(y_hat_1 - y_hat_2) > 0).shape)
        changed = np.abs(y_hat_1 - y_hat_2) > 0
        print(f'{np.sum(changed)} of {np.prod(inp.size())} are changed')
        print(f'sqrt: {np.sqrt(np.sum(changed))}')
        print(f'min, max arg height {np.min(np.argwhere(changed)[:, 2])},{np.max(np.argwhere(changed)[:, 2])}')
        print(f'min, max arg width {np.min(np.argwhere(changed)[:, 3])},{np.max(np.argwhere(changed)[:, 3])}')

    def test_resunet_v2(self):
        torch.manual_seed(0)
        resunet = Resnet50_UNetV2()
        resunet = resunet.eval()
        with torch.no_grad():
            inp = torch.randn([1, 1, 1024, 1024])
            inp[0, 0, 0, 0] = 0

            y_hat_1 = torch.narrow(resunet(inp), 1, 0, 1).numpy()
            inp[0, 0, 0, 0] = 1

            y_hat_2 = torch.narrow(resunet(inp), 1, 0, 1).numpy()
        self.analyze(y_hat_1, y_hat_2, inp, 'Resnet50_UNetV2')

    def test_res_unet_rdb(self):
        torch.manual_seed(0)
        resunet = Resnet50UNetRDB()
        resunet = resunet.eval()
        with torch.no_grad():
            inp = torch.randn([1, 1, 1024, 1024])
            inp[0, 0, 0, 0] = 0

            y_hat_1 = torch.narrow(resunet(inp), 1, 0, 1).numpy()
            inp[0, 0, 0, 0] = 1

            y_hat_2 = torch.narrow(resunet(inp), 1, 0, 1).numpy()
        self.analyze(y_hat_1, y_hat_2, inp, 'Resnet50UNetRDB')

    def test_resnext_unet(self):
        torch.manual_seed(0)
        resunet = Resnext50_UNet()
        resunet = resunet.eval()
        with torch.no_grad():
            inp = torch.randn([1, 1, 1024, 1024])
            inp[0, 0, 0, 0] = 0

            y_hat_1 = torch.narrow(resunet(inp), 1, 0, 1).numpy()
            inp[0, 0, 0, 0] = 1

            y_hat_2 = torch.narrow(resunet(inp), 1, 0, 1).numpy()
        self.analyze(y_hat_1, y_hat_2, inp, 'Resnext50_UNet')

    def test_resnet_unet(self):
        torch.manual_seed(0)
        resunet = Resnet50_UNet()
        resunet = resunet.eval()
        with torch.no_grad():
            inp = torch.randn([1, 1, 1024, 1024])
            inp[0, 0, 0, 0] = 0

            y_hat_1 = torch.narrow(resunet(inp), 1, 0, 1).numpy()
            inp[0, 0, 0, 0] = 1

            y_hat_2 = torch.narrow(resunet(inp), 1, 0, 1).numpy()
        self.analyze(y_hat_1, y_hat_2, inp, 'Resnet50_UNet')

    def test_resnet101_unet(self):
        torch.manual_seed(0)
        resunet = Resnet101_UNetV2()
        resunet = resunet.eval()
        with torch.no_grad():
            inp = torch.randn([1, 1, 1024, 1024])
            inp[0, 0, 0, 0] = 0

            y_hat_1 = torch.narrow(resunet(inp), 1, 0, 1).numpy()
            inp[0, 0, 0, 0] = 1

            y_hat_2 = torch.narrow(resunet(inp), 1, 0, 1).numpy()
        self.analyze(y_hat_1, y_hat_2, inp, 'Resnet101_UNetV2')

    def test_DenseNet121UNetRDB(self):
        torch.manual_seed(0)
        resunet = DenseNet121UNetRDB()
        resunet = resunet.eval()
        with torch.no_grad():
            inp = torch.randn([1, 1, 1024, 1024])
            inp[0, 0, 0, 0] = 0

            y_hat_1 = torch.narrow(resunet(inp), 1, 0, 1).numpy()
            inp[0, 0, 0, 0] = 1

            y_hat_2 = torch.narrow(resunet(inp), 1, 0, 1).numpy()
        self.analyze(y_hat_1, y_hat_2, inp, 'DenseNet121UNetRDB')

    def test_UNetVGG16DualEncoder(self):
        torch.manual_seed(0)
        resunet = UNetVGG16DualEncoder()
        resunet = resunet.eval()
        with torch.no_grad():
            inp = torch.randn([1, 1, 1024, 1024])
            inp[0, 0, 0, 0] = 0

            y_hat_1 = torch.narrow(resunet(inp), 1, 0, 1).numpy()
            inp[0, 0, 0, 0] = 1

            y_hat_2 = torch.narrow(resunet(inp), 1, 0, 1).numpy()
        self.analyze(y_hat_1, y_hat_2, inp, 'UNetVGG16DualEncoder')

    def test_InceptionResNetV2UNetAttention(self):
        torch.manual_seed(0)
        model = InceptionResNetV2UNetAttention()
        model = model.eval()
        with torch.no_grad():
            inp = torch.randn([1, 1, 2048, 2048])
            inp[0, 0, 0, 0] = 0

            y_hat_1 = torch.narrow(model(inp), 1, 0, 1).numpy()
            inp[0, 0, 0, 0] = 1

            y_hat_2 = torch.narrow(model(inp), 1, 0, 1).numpy()
        self.analyze(y_hat_1, y_hat_2, inp, 'InceptionResNetV2UNetAttention')


if __name__ == '__main__':
    test = TestReceptiveField()
    #test.test_resunet_v2()
    #test.test_resnet_unet()
    #test.test_resnext_unet()
    #test.test_res_unet_rdb()
    #test.test_DenseNet121UNetRDB()
    #test.test_resnet101_unet()
    #test.test_UNetVGG16DualEncoder()
    test.test_InceptionResNetV2UNetAttention()
