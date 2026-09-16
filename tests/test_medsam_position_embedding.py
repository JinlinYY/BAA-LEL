import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from bua_lel.models.backbones.medsam import MedSAMMultiScaleEncoder


class _PatchEmbed(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=16).permute(0, 2, 3, 1).contiguous()


class _FakeImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = _PatchEmbed()
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, 64, 3))
        self.blocks = nn.ModuleList([nn.Identity()])
        self.neck = nn.Identity()


class MedSAMPositionEmbeddingTests(unittest.TestCase):
    def test_pretrained_position_embedding_adapts_to_256_input_grid(self):
        wrapper = SimpleNamespace(image_encoder=_FakeImageEncoder(), out_indices=())
        image = torch.randn(2, 3, 256, 256)

        encoded, intermediate = MedSAMMultiScaleEncoder.forward_medsam_with_intermediate(
            wrapper, image
        )

        self.assertEqual(tuple(encoded.shape), (2, 3, 16, 16))
        self.assertEqual(intermediate, {})


if __name__ == "__main__":
    unittest.main()
