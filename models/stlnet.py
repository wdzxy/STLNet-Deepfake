from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn

from .layers import _positive_int
from .sfe import ShallowFeatureExtractor
from .tem import TextureEnhancementModule
from .lcm import LightweightClassifier
from .rep_classifier import RepClassifierBlock


class STLNet(nn.Module):
    def __init__(self, widths: Sequence[int] = (64, 96, 128),
                 depths: Sequence[int] = (2, 2, 2),
                 expansion: float | Sequence[float] = (2.0, 47.0 / 3.0, 39.0),
                 stem_width: int = 32, attention_heads: int = 4,
                 classifier_width: int = 256, adapter_width: int = 128,
                 shift_stages: Sequence[bool] = (False, True, False),
                 use_adapter: bool = True,
                 deploy: bool = False,
                 stem_strides: Sequence[int] = (2, 2, 2)) -> None:
        super().__init__()
        if len(widths) != 3 or len(depths) != 3 or len(shift_stages) != 3:
            raise ValueError("widths, depths, and shift_stages must each contain three values")
        widths = tuple(_positive_int("widths", c) for c in widths)
        depths = tuple(_positive_int("depths", d) for d in depths)
        for name, value in (("stem_width", stem_width), ("attention_heads", attention_heads),
                            ("classifier_width", classifier_width), ("adapter_width", adapter_width)):
            _positive_int(name, value)
        if isinstance(expansion, (int, float)) and not isinstance(expansion, bool):
            expansion = (expansion,) * 3
        if not isinstance(expansion, (list, tuple)) or len(expansion) != 3:
            raise ValueError("expansion must be a positive scalar or three positive ratios")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
               not math.isfinite(v) or v <= 0 for v in expansion):
            raise ValueError("every expansion ratio must be finite and positive")
        expansion = tuple(float(v) for v in expansion)
        if not isinstance(stem_strides, (list, tuple)) or not stem_strides:
            raise ValueError("stem_strides must contain at least one positive stride")
        stem_strides = tuple(_positive_int("stem_strides", v) for v in stem_strides)
        if not all(isinstance(v, bool) for v in shift_stages):
            raise ValueError("shift_stages values must be booleans")
        if not isinstance(deploy, bool) or not isinstance(use_adapter, bool):
            raise ValueError("deploy and use_adapter must be booleans")
        self.model_config = {
            "widths": list(widths), "depths": list(depths), "expansion": list(expansion),
            "stem_width": stem_width, "attention_heads": attention_heads,
            "classifier_width": classifier_width, "adapter_width": adapter_width,
            "shift_stages": list(shift_stages),
            "use_adapter": use_adapter, "deploy": deploy, "stem_strides": list(stem_strides),
        }
        self.deploy = deploy
        self.sfe = ShallowFeatureExtractor(widths, depths, expansion, stem_width, stem_strides,
                                          shift_stages)
        self.tem = TextureEnhancementModule(widths, attention_heads)
        self.descriptor_dim = sum(widths) * attention_heads
        self.lcm = LightweightClassifier(self.descriptor_dim, classifier_width,
                                         adapter_width, use_adapter, deploy)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_features(self, x: Tensor) -> dict:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Expected RGB input with shape [batch, 3, height, width]")
        if x.shape[0] == 0 or min(x.shape[2:]) < 16:
            raise ValueError("Input must have a nonempty batch and height/width >= 16")
        shallow = self.sfe(x)
        features = self.tem(shallow)
        features.update({
            "sfe": shallow, "sfe_features": shallow, "shallow_features": shallow,
            "texture_features": features["textures"], "attention_maps": features["attention"],
        })
        features["embedding"] = self.lcm.embed(features["descriptor"])
        return features

    def forward(self, x: Tensor, return_features: bool = False):
        features = self.forward_features(x)
        logits = self.lcm.output(features["embedding"]).squeeze(1)
        return (logits, features) if return_features else logits

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.training:
            raise RuntimeError("Call model.eval() before switch_to_deploy().")
        for module in self.modules():
            if isinstance(module, RepClassifierBlock):
                module.switch_to_deploy()
        self.deploy = True
        self.model_config["deploy"] = True
        return self
