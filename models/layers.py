from torch import Tensor, nn


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class ConvBNGELU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3,
                 stride: int = 1, groups: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel, stride=stride,
                      padding=kernel // 2, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
