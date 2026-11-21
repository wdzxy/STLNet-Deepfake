from torch import Tensor, nn
from torch.nn import functional as F


class ResidualAdapter(nn.Module):
    def __init__(self, channels: int = 256, bottleneck: int = 128) -> None:
        super().__init__()
        self.down = nn.Conv2d(channels, bottleneck, 1, bias=True)
        self.up = nn.Conv2d(bottleneck, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.bn(self.up(F.relu(self.down(x))))
