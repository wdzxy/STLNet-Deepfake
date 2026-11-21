from typing import Sequence

from torch import Tensor, nn

from .layers import ConvBNGELU
from .lstb import LSTB
from .token_shift import TokenShift


class ShallowFeatureExtractor(nn.Module):
    def __init__(self, widths: Sequence[int], depths: Sequence[int],
                 expansion: Sequence[float], stem_width: int,
                 stem_strides: Sequence[int],
                 shift_stages: Sequence[bool]) -> None:
        super().__init__()
        stem_channels = [3] + [stem_width] * (len(stem_strides) - 1) + [widths[0]]
        self.stem = nn.Sequential(*[
            ConvBNGELU(stem_channels[i], stem_channels[i + 1], 3, stride)
            for i, stride in enumerate(stem_strides)
        ])
        self.transitions = nn.ModuleList([
            nn.Identity(),
            ConvBNGELU(widths[0], widths[1], 3, 2),
            ConvBNGELU(widths[1], widths[2], 3, 2),
        ])
        self.shifts = nn.ModuleList([
            TokenShift() if enabled else nn.Identity()
            for enabled in shift_stages
        ])
        self.stages = nn.ModuleList([
            nn.Sequential(*[
                LSTB(channels, expansion[stage], pre_norm=True)
                for _ in range(depth)
            ])
            for stage, (channels, depth) in enumerate(zip(widths, depths))
        ])

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.stem(x)
        outputs = []
        for shift, transition, stage in zip(self.shifts, self.transitions, self.stages):
            x = stage(transition(shift(x)))
            outputs.append(x)
        return tuple(outputs)
