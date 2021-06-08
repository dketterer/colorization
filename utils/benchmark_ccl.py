import timeit

import torch

from loss import ColorConsistencyLoss
B = 128
C = 2
H, W = 256, 256
S = 20
N = 1000
version = 'linear'

ab = torch.rand(B, C, H, W).cuda()
segment_masks = torch.zeros(S, H, W, dtype=torch.bool).reshape(-1)
segment_masks[torch.randint(len(segment_masks), (H * W,))] = 1
segment_masks = segment_masks.reshape(1, S, H, W)
segment_masks = segment_masks.expand(B, S, H, W).cuda()

best_ccl = ColorConsistencyLoss(version).cuda()


def no_for_ccl():
    best_ccl(ab, segment_masks)


if __name__ == '__main__':
    print(timeit.timeit('no_for_ccl()', globals=locals(), number=N) / N)  # 28s
