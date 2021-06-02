import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, BoolTensor, Tensor, LongTensor
from torch.cuda.amp import autocast
from torch.nn import MSELoss, Parameter
from torch.nn import SmoothL1Loss as TorchL1Loss

Q_PRIOR = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../resources/q-prior.npy'))
AB_GAMUT = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../resources/ab-gamut.npy'))


class ColorConsistencyLoss(nn.Module):
    """
    in HSV color space, this loss would only touch the hue
    in RGB it touches all channels
    in Lab it uses a and b

    """

    def __init__(self, mode='linear'):
        super(ColorConsistencyLoss, self).__init__()
        assert mode in ['square', 'euclidean', 'linear']
        self.mode = mode

    def forward(self, x: Tensor, masks: Tensor) -> Tensor:
        inp_dtype = x.dtype
        # x: (B, C, H, W)
        # masks: (B, H, W) LongTensor
        masks.requires_grad = False
        # masks: (B, S, H, W)
        masks = F.one_hot(masks).bool().permute((0, 3, 1, 2)).contiguous()

        # expand x and the masks to make the loss calculation vectorized per image
        x_exp = torch.unsqueeze(x, 2).expand(x.size()[0:2] + (len(masks[0]),) + x.size()[2:])
        batch_segment_masks_exp = torch.unsqueeze(masks, 1).expand(
            x.size()[0:2] + (len(masks[0]),) + x.size()[2:]).contiguous()
        batch_segment_masks_exp.requires_grad = False

        # parallel for all in batch dim
        # for each channel (a, b)

        masked = torch.mul(x_exp, batch_segment_masks_exp)  # (B, C, S, H, W)
        masked = masked.detach()
        x_mean = masked.sum(-1).sum(-1) / (batch_segment_masks_exp.sum(-1).sum(-1) + 1e-8)  # (B, C, S)
        # (B, C, S, H, W) all values in a segment are the same
        x_mean = torch.unsqueeze(torch.unsqueeze(x_mean, -1), -1).expand_as(x_exp)
        x_mean = x_mean.detach()

        if self.mode == 'square':
            # inputs to mse: (B, C, len(segment))
            with autocast(enabled=False):
                loss = F.mse_loss(x_exp[batch_segment_masks_exp].float(),
                                  x_mean[batch_segment_masks_exp].float()).type(inp_dtype)  # scalar, can be nan
        elif self.mode == 'euclidean':
            # inputs to square: (B, C, len(segment))
            squared = torch.square(x_exp[batch_segment_masks_exp] -
                                   x_mean[batch_segment_masks_exp])  # (B, C, S, len(segment))
            summed = squared.reshape(x.size()[0:2] + (-1,)).sum(1)
            # gradient of sqrt(0) is nan, eps is required
            loss = torch.sqrt_(summed + 1e-8).mean()
        elif self.mode == 'linear':
            # inputs to L1: (B, C, len(segment))
            with autocast(enabled=False):
                loss = F.smooth_l1_loss(x_exp[batch_segment_masks_exp].float(),
                                        x_mean[batch_segment_masks_exp].float()).type(inp_dtype)  # scalar, can be nan
        if torch.isnan(loss):
            return torch.tensor(0.0)

        return loss


class L1CCLoss(nn.Module):
    def __init__(self, lambda_ccl, ccl_version, weighted=False, alpha=5, gamma=.5):
        super().__init__()
        self.lambda_ccl = lambda_ccl
        self.ccl_version = ccl_version
        self.ccl = ColorConsistencyLoss(self.ccl_version)
        self.l1 = L1Loss(weighted, alpha=alpha, gamma=gamma)

    def forward(self, input: Tensor, target: Tensor, segment_masks: LongTensor):
        ccl_loss = self.ccl(input, segment_masks)
        l1_loss = self.l1(input, target)[0]
        loss = l1_loss + ccl_loss * self.lambda_ccl
        return loss, {'L1CCLoss': loss, f'ColorConsistencyLoss-{self.ccl_version}': ccl_loss,
                      'L1Loss': l1_loss}


class L2CCLoss(nn.Module):
    def __init__(self, lambda_ccl, ccl_version, weighted=False, alpha=5, gamma=.5):
        super().__init__()
        self.lambda_ccl = lambda_ccl
        self.ccl_version = ccl_version
        self.ccl = ColorConsistencyLoss(self.ccl_version)
        self.l2 = L2Loss(weighted, alpha=alpha, gamma=gamma)

    def forward(self, input: Tensor, target: Tensor, segment_masks: LongTensor):
        ccl_loss = self.ccl(input, segment_masks)
        l2_loss = self.l2(input, target)[0]
        loss = l2_loss + ccl_loss * self.lambda_ccl
        return loss, {'L2CCLoss': loss, f'ColorConsistencyLoss-{self.ccl_version}': ccl_loss,
                      'MSE': l2_loss}


class L2Loss(nn.Module):
    """
    Wrapper to accept 3 args in the forward pass
    """

    def __init__(self, weighted=False, alpha=5, gamma=.5):
        super().__init__()
        self.weighted = weighted
        if self.weighted:
            self.prior_weights = PriorWeights(alpha=alpha, gamma=gamma)

    def forward(self, input: Tensor, target: Tensor, segment_masks: Optional[BoolTensor] = None):
        inp_type = input.dtype
        with autocast(enabled=False):
            l2_loss = MSELoss(reduction='none')(input.float(), target.float()).type(inp_type)
        if self.weighted:
            weights_implied = self.prior_weights(target)
            l2_loss = (l2_loss * weights_implied)
        l2_loss = l2_loss.sum((1, 2, 3)).mean()
        return l2_loss, {'MSE': l2_loss}


class L1Loss(nn.Module):
    """
    Wrapper to accept 3 args in the forward pass
    """

    def __init__(self, weighted=False, alpha=5, gamma=.5):
        super().__init__()
        self.weighted = weighted
        if self.weighted:
            self.prior_weights = PriorWeights(alpha=alpha, gamma=gamma)

    def forward(self, input: Tensor, target: Tensor, segment_masks: Optional[BoolTensor] = None):
        inp_type = input.dtype
        with autocast(enabled=False):
            l1_loss = TorchL1Loss(reduction='none')(input.float(), target.float()).type(inp_type)
        if self.weighted:
            weights_implied = self.prior_weights(target)
            l1_loss = (l1_loss * weights_implied)
        l1_loss = l1_loss.sum((1, 2, 3)).mean()
        return l1_loss, {'L1Loss': l1_loss}


class PriorWeights(nn.Module):
    def __init__(self, alpha=5, gamma=.5):
        super(PriorWeights, self).__init__()
        prior_probs = Q_PRIOR

        # define uniform probability
        uni_probs = np.zeros_like(prior_probs)
        uni_probs[prior_probs != 0] = 1.
        uni_probs = uni_probs / np.sum(uni_probs)

        # convex combination of empirical prior and uniform distribution
        prior_mix = (1 - gamma) * prior_probs + gamma * uni_probs

        # set prior factor
        prior_factor = prior_mix ** -alpha
        prior_factor = prior_factor / np.sum(prior_probs * prior_factor)  # re-normalize

        # implied empirical prior
        implied_prior = prior_probs * prior_factor
        implied_prior = implied_prior / np.sum(implied_prior)  # re-normalize
        self.implied_prior = Parameter(Parameter(torch.from_numpy(implied_prior).float()), requires_grad=False)
        self.ab_gamut = Parameter(((torch.from_numpy(AB_GAMUT).float() + 128) / 128) - 1, requires_grad=False)

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, target):
        b, c, h, w = target.size()
        target = target.view(target.size()[0], target.size()[1], -1)
        target = torch.transpose(target, 2, 1)
        cdist = torch.cdist(self.ab_gamut, target)
        nns = cdist.argmin(1)
        weights_implied = self.implied_prior[nns].reshape(b, 1, h, w).expand(b, c, h, w)
        return weights_implied
