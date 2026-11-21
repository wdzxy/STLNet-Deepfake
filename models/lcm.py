from torch import Tensor, nn
from torch.nn import functional as F

from .adapter import ResidualAdapter
from .rep_classifier import RepClassifierBlock


class LightweightClassifier(nn.Module):
    def __init__(self, in_features: int = 1152, width: int = 256,
                 adapter_width: int = 128, use_adapter: bool = True,
                 deploy: bool = False) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_features, width, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )
        self.rep_blocks = nn.Sequential(
            RepClassifierBlock(width, deploy), RepClassifierBlock(width, deploy))
        self.adapter = ResidualAdapter(width, adapter_width) if use_adapter else nn.Identity()
        self.output = nn.Linear(width, 1)

    def embed(self, descriptor: Tensor) -> Tensor:
        h = self.projection(descriptor[:, :, None, None])
        h = self.adapter(self.rep_blocks(h))
        return F.adaptive_avg_pool2d(h, 1).flatten(1)

    def forward(self, descriptor: Tensor) -> Tensor:
        return self.output(self.embed(descriptor)).squeeze(1)
