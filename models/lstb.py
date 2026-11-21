from torch import Tensor, nn

from .layers import ChannelLayerNorm, ConvBNGELU
from .token_shift import TokenShift


class LSTB(nn.Module):
    def __init__(self, channels: int, expansion: float = 2.0,
                 pre_norm: bool = True) -> None:
        super().__init__()
        hidden = max(1, round(channels * expansion))
        self.pre_norm = ChannelLayerNorm(channels) if pre_norm else nn.Identity()
        self.token_shift = TokenShift()
        self.local_mix = nn.Sequential(
            ConvBNGELU(channels, channels, 3, groups=channels),
            ConvBNGELU(channels, channels, 1),
        )
        self.channel_norm = ChannelLayerNorm(channels) if pre_norm else nn.Identity()
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        mixed = self.local_mix(self.token_shift(self.pre_norm(x)))
        inner = mixed + self.channel_mlp(self.channel_norm(mixed))
        return x + inner
