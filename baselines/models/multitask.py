"""Task-faithful multi-task baselines.

Only methods whose original formulation jointly predicts a mask and a class are
registered here.  This prevents accidental expansion of a single-task baseline
into a different task merely to fill a comparison table.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.models.segmentation import ConvBlock


class MTANet(nn.Module):
    """Independent paper-informed MTANet port for image-clinical diagnosis.

    It uses a shared encoder, reverse-addition segmentation attention and an
    image-clinical attention bottleneck.  The exact upstream implementation was
    not publicly verified, so this module is explicitly provenance-labelled as
    a paper reimplementation rather than author code.
    """
    def __init__(self, clinical_dim: int, num_classes: int = 3, in_channels: int = 3, base_channels: int = 32, **_: object):
        super().__init__()
        if clinical_dim < 1:
            raise ValueError("MTANet requires clinical features for its attention bottleneck")
        c = int(base_channels)
        self.enc0 = ConvBlock(in_channels, c)
        self.enc1 = ConvBlock(c, 2 * c)
        self.enc2 = ConvBlock(2 * c, 4 * c)
        self.bottleneck = ConvBlock(4 * c, 8 * c)
        self.pool = nn.MaxPool2d(2)
        self.coarse_head = nn.Conv2d(8 * c, 1, 1)
        self.dec2 = ConvBlock(8 * c + 4 * c, 4 * c)
        self.dec1 = ConvBlock(4 * c + 2 * c, 2 * c)
        self.dec0 = ConvBlock(2 * c + c, c)
        self.seg_head = nn.Conv2d(c, 1, 1)
        self.image_projection = nn.Sequential(nn.Linear(8 * c, 4 * c), nn.LayerNorm(4 * c), nn.ReLU(True))
        self.clinical_projection = nn.Sequential(nn.Linear(clinical_dim, 4 * c), nn.LayerNorm(4 * c), nn.ReLU(True))
        self.bottleneck_attention = nn.MultiheadAttention(4 * c, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(4 * c, 2 * c), nn.ReLU(True), nn.Dropout(.3), nn.Linear(2 * c, num_classes))

    def forward(self, image: torch.Tensor, clinical: torch.Tensor) -> dict[str, torch.Tensor]:
        enc0 = self.enc0(image)
        enc1 = self.enc1(self.pool(enc0))
        enc2 = self.enc2(self.pool(enc1))
        bottleneck = self.bottleneck(self.pool(enc2))
        coarse = self.coarse_head(bottleneck)

        def reverse_attention(feature: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
            reverse = 1.0 - torch.sigmoid(F.interpolate(reference, feature.shape[-2:], mode="bilinear", align_corners=False))
            return feature * reverse

        decoded = self.dec2(torch.cat([
            F.interpolate(bottleneck, enc2.shape[-2:], mode="bilinear", align_corners=False),
            enc2 + reverse_attention(enc2, coarse),
        ], 1))
        decoded = self.dec1(torch.cat([
            F.interpolate(decoded, enc1.shape[-2:], mode="bilinear", align_corners=False),
            enc1 + reverse_attention(enc1, coarse),
        ], 1))
        decoded = self.dec0(torch.cat([
            F.interpolate(decoded, enc0.shape[-2:], mode="bilinear", align_corners=False),
            enc0 + reverse_attention(enc0, coarse),
        ], 1))
        seg_logits = self.seg_head(decoded)

        image_token = self.image_projection(F.adaptive_avg_pool2d(bottleneck, 1).flatten(1))
        clinical_token = self.clinical_projection(clinical)
        tokens = torch.stack([image_token, clinical_token], 1)
        fused, attention = self.bottleneck_attention(tokens, tokens, tokens, need_weights=True)
        return {
            "segmentation_logits": seg_logits,
            "classification_logits": self.classifier(fused.mean(1)),
            "coarse_logits": F.interpolate(coarse, image.shape[-2:], mode="bilinear", align_corners=False),
            "attention": attention,
        }
