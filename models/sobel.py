import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SobelGradient(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        kernel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        self.register_buffer("kernel_x", kernel[None, None].repeat(channels, 1, 1, 1))
        self.register_buffer("kernel_y", kernel.t()[None, None].repeat(channels, 1, 1, 1))
        self.channels = channels
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        gx = F.conv2d(work, self.kernel_x.to(dtype=work.dtype),
                      padding=1, groups=self.channels)
        gy = F.conv2d(work, self.kernel_y.to(dtype=work.dtype),
                      padding=1, groups=self.channels)
        return (gx.square() + gy.square() + self.eps).sqrt().to(dtype=x.dtype)
