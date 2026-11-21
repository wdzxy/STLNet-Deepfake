import torch
from torch import Tensor, nn


class TokenShift(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        group = x.shape[1] // 4
        if group == 0:
            return x
        out = torch.zeros_like(x)
        out[:, :group, :-1, :] = x[:, :group, 1:, :]
        out[:, group:2 * group, 1:, :] = x[:, group:2 * group, :-1, :]
        out[:, 2 * group:3 * group, :, :-1] = x[:, 2 * group:3 * group, :, 1:]
        out[:, 3 * group:4 * group, :, 1:] = x[:, 3 * group:4 * group, :, :-1]
        out[:, 4 * group:] = x[:, 4 * group:]
        return out
