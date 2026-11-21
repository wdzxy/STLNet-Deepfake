import torch
from torch import Tensor, nn

from .layers import ConvBNGELU
from .sobel import SobelGradient


class TextureResponseUnit(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.sobel = SobelGradient(channels)
        self.branches = nn.ModuleList([
            nn.Sequential(
                ConvBNGELU(channels, channels, kernel, groups=channels),
                ConvBNGELU(channels, channels, 1),
            ) for kernel in (3, 5, 7)
        ])
        self.fusion = ConvBNGELU(channels * 5, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        responses = [x, self.sobel(x)] + [branch(x) for branch in self.branches]
        return self.fusion(torch.cat(responses, dim=1))
