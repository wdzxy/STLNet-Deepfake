import torch
from torch import Tensor, nn


class RepClassifierBlock(nn.Module):
    def __init__(self, channels: int = 256, deploy: bool = False) -> None:
        super().__init__()
        self.channels = channels
        self.deploy = deploy
        self.activation = nn.ReLU()
        if deploy:
            self.reparam = nn.Conv2d(channels, channels, 1, bias=True)
        else:
            self.conv = nn.Conv2d(channels, channels, 1, bias=False)
            self.conv_bn = nn.BatchNorm2d(channels)
            self.identity_bn = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        if self.deploy:
            return self.activation(self.reparam(x))
        return self.activation(self.conv_bn(self.conv(x)) + self.identity_bn(x))

    @staticmethod
    def _fold(kernel: Tensor, bias: Tensor, bn: nn.BatchNorm2d) -> tuple[Tensor, Tensor]:
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        return kernel * scale[:, None, None, None], bn.bias + (bias - bn.running_mean) * scale

    def equivalent_kernel_bias(self) -> tuple[Tensor, Tensor]:
        if self.deploy:
            return self.reparam.weight, self.reparam.bias
        zeros = self.conv.weight.new_zeros(self.channels)
        conv_kernel, conv_bias = self._fold(self.conv.weight, zeros, self.conv_bn)
        identity = torch.eye(self.channels, dtype=conv_kernel.dtype,
                             device=conv_kernel.device).reshape(self.channels, self.channels, 1, 1)
        identity_kernel, identity_bias = self._fold(identity, zeros, self.identity_bn)
        return conv_kernel + identity_kernel, conv_bias + identity_bias

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.training:
            raise RuntimeError("Call eval() before fusing RepClassifier BatchNorm branches.")
        if self.deploy:
            return self
        kernel, bias = self.equivalent_kernel_bias()
        self.reparam = nn.Conv2d(self.channels, self.channels, 1, bias=True).to(
            device=kernel.device, dtype=kernel.dtype)
        self.reparam.weight.copy_(kernel)
        self.reparam.bias.copy_(bias)
        self.reparam.eval()
        del self.conv, self.conv_bn, self.identity_bn
        self.deploy = True
        return self
