import torch
from torch import Tensor, nn


class AttentionPoolingUnit(nn.Module):
    def __init__(self, channels: int, heads: int = 4, eps: float = 1e-6) -> None:
        super().__init__()
        self.scores = nn.Conv2d(channels, heads, 1, bias=True)
        self.eps = eps

    def forward(self, x: Tensor, return_attention: bool = False):
        scores = self.scores(x).flatten(2)
        work_dtype = torch.float32 if scores.dtype in (torch.float16, torch.bfloat16) else scores.dtype
        attention = scores.to(work_dtype).softmax(dim=-1)
        pooled = torch.bmm(attention, x.flatten(2).transpose(1, 2).to(work_dtype))
        pooled = pooled / (attention.sum(dim=-1, keepdim=True) + self.eps)
        descriptor = pooled.flatten(1).to(x.dtype)
        if return_attention:
            return descriptor, attention.reshape(x.shape[0], -1, x.shape[2], x.shape[3])
        return descriptor
