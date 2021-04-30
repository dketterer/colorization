import warnings
from typing import List, Union, Optional

import torch
import torch.nn.functional as F
from torch import nn, BoolTensor, Tensor
from torch.nn import MSELoss
from torch.nn import SmoothL1Loss as TorchL1Loss


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

    # Hack to fix this issue https://github.com/pytorch/pytorch/issues/38487
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, x: Tensor, masks: BoolTensor) -> Tensor:
        # x: (B, C, H, W)
        # masks: (B, S, H, W)
        # there must be exactly B*C*H*W 'True' values in the masks

        # expand x and the masks to make the loss calculation vectorized per image
        x_exp = torch.unsqueeze(x, 2).expand(x.size()[0:2] + (len(masks[0]),) + x.size()[2:])
        batch_segment_masks_exp = torch.unsqueeze(masks, 1).expand(
            x.size()[0:2] + (len(masks[0]),) + x.size()[2:])

        # parallel for all in batch dim
        # for each channel (a, b)

        masked = torch.mul(x_exp, batch_segment_masks_exp)  # (B, C, S, H, W)
        x_mean = masked.sum(-1).sum(-1) / (batch_segment_masks_exp.sum(-1).sum(-1) + 1e-8)  # (B, C,S,len(segment_masks)
        # (B, C, S, H, W) all values in a segment are the same
        x_mean = torch.unsqueeze(torch.unsqueeze(x_mean, -1), -1).expand_as(x_exp)

        if self.mode == 'square':
            # inputs to mse: (B, C, len(segment))
            loss = F.mse_loss(x_exp[batch_segment_masks_exp], x_mean[batch_segment_masks_exp])  # scalar, can be nan
        elif self.mode == 'euclidean':
            # inputs to square: (B, C, len(segment))
            squared = torch.square(x_exp[batch_segment_masks_exp] -
                                   x_mean[batch_segment_masks_exp])  # (B, C, S, len(segment))
            summed = squared.reshape(x.size()[0:2] + (-1,)).sum(1)
            # gradient of sqrt(0) is nan, eps is required
            loss = torch.sqrt_(summed + 1e-8).mean()
        elif self.mode == 'linear':
            # inputs to L1: (B, C, len(segment))
            loss = F.smooth_l1_loss(x_exp[batch_segment_masks_exp],
                                    x_mean[batch_segment_masks_exp])  # scalar, can be nan
        if torch.isnan(loss):
            return torch.tensor(0.0)

        return loss


class L1CCLoss(nn.Module):
    def __init__(self, lambda_ccl, ccl_version):
        super().__init__()
        self.lambda_ccl = lambda_ccl
        self.ccl_version = ccl_version
        self.ccl = ColorConsistencyLoss(self.ccl_version)

    # Hack to fix this issue https://github.com/pytorch/pytorch/issues/38487
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, input: Tensor, target: Tensor, segment_masks: BoolTensor):
        ccl_loss = self.ccl(input, segment_masks)
        l1_loss = TorchL1Loss()(input, target)
        loss = l1_loss + ccl_loss * self.lambda_ccl
        return loss, {'L1CCLoss': loss, f'ColorConsistencyLoss-{self.ccl_version}': ccl_loss,
                      'L1Loss': l1_loss}


class L2CCLoss(nn.Module):
    def __init__(self, lambda_ccl, ccl_version):
        super().__init__()
        self.lambda_ccl = lambda_ccl
        self.ccl_version = ccl_version
        self.ccl = ColorConsistencyLoss(self.ccl_version)

    # Hack to fix this issue https://github.com/pytorch/pytorch/issues/38487
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, input: Tensor, target: Tensor, segment_masks: BoolTensor):
        ccl_loss = self.ccl(input, segment_masks)
        l2_loss = MSELoss()(input, target)
        loss = l2_loss + ccl_loss * self.lambda_ccl
        return loss, {'L2CCLoss': loss, f'ColorConsistencyLoss-{self.ccl_version}': ccl_loss,
                      'MSE': l2_loss}


class L2Loss(nn.Module):
    """
    Wrapper to accept 3 args in the forward pass
    """

    def __init__(self):
        super().__init__()

    # Hack to fix this issue https://github.com/pytorch/pytorch/issues/38487
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, input: Tensor, target: Tensor, segment_masks: Optional[BoolTensor] = None):
        l2_loss = MSELoss()(input, target)
        return l2_loss, {'MSE': l2_loss}


class L1Loss(nn.Module):
    """
    Wrapper to accept 3 args in the forward pass
    """

    def __init__(self):
        super().__init__()

    # Hack to fix this issue https://github.com/pytorch/pytorch/issues/38487
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, input: Tensor, target: Tensor, segment_masks: Optional[BoolTensor] = None):
        l1_loss = TorchL1Loss()(input, target)
        return l1_loss, {'L1Loss': l1_loss}
