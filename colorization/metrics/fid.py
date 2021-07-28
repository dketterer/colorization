# https://github.com/hukkelas/pytorch-frechet-inception-distance/blob/master/fid.py

import scipy
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import inception_v3
from scipy import linalg


def to_cuda(elements):
    """
    Transfers elements to cuda if GPU is available
    Args:
        elements: torch.tensor or torch.nn.module
        --
    Returns:
        elements: same as input on GPU memory, if available
    """
    if torch.cuda.is_available():
        return elements.cuda()
    return elements


class PartialInceptionNetwork(nn.Module):

    def __init__(self, transform_input=True):
        super().__init__()
        self.inception_network = inception_v3(pretrained=True)
        self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)
        self.transform_input = transform_input

    def output_hook(self, module, input, output):
        # N x 2048 x 8 x 8
        self.mixed_7c_output = output

    def forward(self, x):
        """
        Args:
            x: shape (N, 3, 299, 299) dtype: torch.float32 in range 0-1
        Returns:
            inception activations: torch.tensor, shape: (N, 2048), dtype: torch.float32
        """
        assert x.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" + \
                                             ", but got {}".format(x.shape)
        x = x * 2 - 1  # Normalize to [-1, 1]

        # Trigger output hook
        with torch.no_grad():
            self.inception_network(x)

        # Output: N x 2048 x 1 x 1
        activations = self.mixed_7c_output
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1, 1))
        activations = activations.view(x.shape[0], 2048)
        return activations


class FrechetInceptionDistance(nn.Module):
    def __init__(self, num_images):
        super(FrechetInceptionDistance, self).__init__()
        self.inception_network = PartialInceptionNetwork()
        self.inception_network = to_cuda(self.inception_network)
        self.inception_network.eval()
        self.activations1 = torch.empty((num_images, 2048))
        self.activations2 = torch.empty((num_images, 2048))
        self.counter1 = 0
        self.counter2 = 0

    def preprocess_images(self, images):
        """Resizes and shifts the dynamic range of image to 0-1
        Args:
            im: torch.tensor, shape: (B, H, W, 3), dtype: float32 between 0-1 or np.uint8
        Return:
            im: torch.tensor, shape: (B, 3, 299, 299), dtype: torch.float32 between 0-1
        """
        assert images.size()[-1] == 3
        assert len(images.size()) == 4
        if images.dtype == torch.uint8:
            images = images.float() / 255
        images = images.permute(0, 3, 1, 2)
        images = F.interpolate(images, (299, 299), mode='bilinear', align_corners=True)
        assert images.max() <= 1.0
        assert images.min() >= 0.0
        assert images.dtype == torch.float32
        assert images.size()[1:] == (3, 299, 299)

        return images

    def calc_activations1(self, images):
        images = to_cuda(images)
        images = self.preprocess_images(images)
        batch_size = images.size()[0]
        with torch.no_grad():
            act = self.inception_network(images).detach().cpu()
        self.activations1[self.counter1:self.counter1 + batch_size, :] = act
        self.counter1 += batch_size

    def calc_activations2(self, images):
        images = to_cuda(images)
        images = self.preprocess_images(images)
        batch_size = images.size()[0]
        with torch.no_grad():
            act = self.inception_network(images).detach().cpu()
        self.activations2[self.counter2:self.counter2 + batch_size, :] = act
        self.counter2 += batch_size

    def calculate_activation_statistics(self, act):
        """Calculates the statistics used by FID
        Args:
            images: torch.tensor, shape: (N, 3, H, W), dtype: torch.float32 in range 0 - 1
            batch_size: batch size to use to calculate inception scores
        Returns:
            mu:     mean over all activations from the last pool layer of the inception model
            sigma:  covariance matrix over all activations from the last pool layer
                    of the inception model.
        """

        def cov(X):
            D = X.shape[-1]
            mean = torch.mean(X, dim=-1).unsqueeze(-1)
            X = X - mean
            return 1 / (D - 1) * X @ X.transpose(-1, -2)

        mu = torch.mean(act, dim=0)
        sigma = cov(act)
        return mu, sigma

    # Modified from: https://github.com/bioinf-jku/TTUR/blob/master/fid.py
    def calculate_frechet_distance(self, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1 : Numpy array containing the activations of the pool_3 layer of the
                 inception net ( like returned by the function 'get_predictions')
                 for generated samples.
        -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
                   on an representive data set.
        -- sigma1: The covariance matrix over activations of the pool_3 layer for
                   generated samples.
        -- sigma2: The covariance matrix over activations of the pool_3 layer,
                   precalcualted on an representive data set.
        Returns:
        --   : The Frechet Distance.
        """

        # mu1 = np.atleast_1d(mu1)
        # mu2 = np.atleast_1d(mu2)

        # sigma1 = np.atleast_2d(sigma1)
        # sigma2 = np.atleast_2d(sigma2)

        mu1, sigma1 = self.calculate_activation_statistics(self.activations1)
        mu2, sigma2 = self.calculate_activation_statistics(self.activations2)

        assert mu1.size() == mu2.size(), "Training and test mean vectors have different lengths"
        assert sigma1.size() == sigma2.size(), "Training and test covariances have different dimensions"

        diff = mu1 - mu2
        # product might be almost singular
        covmean = torch.from_numpy(
            scipy.linalg.sqrtm(A=(sigma1 @ sigma2).numpy(), disp=False)[0]
        ).float()
        if not torch.isfinite(covmean).all():
            offset = torch.eye(sigma1.shape[0]) * eps
            covmean = torch.from_numpy(
                scipy.linalg.sqrtm(A=((sigma1 + offset) @ sigma2 + offset).numpy(), disp=False)[0]
            ).float()

        if torch.is_complex(covmean):
            if not torch.allclose(torch.diagonal(covmean).imag, torch.tensor(0.0), atol=1e-3):
                m = torch.max(torch.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real.float()

        return diff @ diff + torch.trace(sigma1 + sigma2 - 2 * covmean)
