from typing import Sequence

import torch
from torch import Tensor, nn

from .tru import TextureResponseUnit
from .apu import AttentionPoolingUnit


class TextureEnhancementModule(nn.Module):
    def __init__(self, widths: Sequence[int], heads: int = 4) -> None:
        super().__init__()
        self.trus = nn.ModuleList([TextureResponseUnit(c) for c in widths])
        self.apus = nn.ModuleList([AttentionPoolingUnit(c, heads) for c in widths])

    def forward(self, features: Sequence[Tensor]) -> dict:
        textures, descriptors, attention_maps = [], [], []
        for feature, tru, apu in zip(features, self.trus, self.apus):
            texture = tru(feature)
            descriptor, attention = apu(texture, return_attention=True)
            textures.append(texture)
            descriptors.append(descriptor)
            attention_maps.append(attention)
        return {
            "textures": tuple(textures),
            "stage_descriptors": tuple(descriptors),
            "attention": tuple(attention_maps),
            "descriptor": torch.cat(descriptors, dim=1),
        }
