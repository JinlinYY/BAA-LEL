"""ImageNet ResNet-50 adapter for the shared BAA-LEL encoder contract."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from baa_lel.models.backbones.medsam import ConvGNAct


class ResNet50MultiScaleEncoder(nn.Module):
    """Expose ResNet-50 features in the same shape contract as MedSAM.

    The ImageNet backbone is frozen except for its final bottleneck. Projection
    and fusion layers are task-specific and remain trainable.
    """

    def __init__(
        self,
        checkpoint_path=None,
        pretrained: bool = True,
        unfreeze_last_bottleneck: bool = True,
        out_channels: Tuple[int, int, int, int] = (120, 240, 480, 960),
        cls_out_dim: int = 256,
        gn_groups: int = 32,
        **_: object,
    ):
        super().__init__()
        from torchvision.models import resnet50

        network = resnet50(weights=None)
        if checkpoint_path:
            state = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
            state = state.get("state_dict", state.get("model_state", state))
            network.load_state_dict(state, strict=True)
        elif pretrained:
            from torchvision.models import ResNet50_Weights

            network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        self.image_encoder = network
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = False
        if unfreeze_last_bottleneck:
            for parameter in self.image_encoder.layer4[-1].parameters():
                parameter.requires_grad = True

        c1, c2, c3, c4 = out_channels
        self.f1_adapter = ConvGNAct(64, c1, k=3, gn_groups=gn_groups)
        self.f2_adapter = ConvGNAct(256, c2, k=3, gn_groups=gn_groups)
        self.f3_adapter = ConvGNAct(512, c3, k=3, gn_groups=gn_groups)
        self.f4_adapter = ConvGNAct(2048, c4, k=3, gn_groups=gn_groups)

        self.cls_layer2 = ConvGNAct(512, cls_out_dim, k=1, p=0, gn_groups=gn_groups)
        self.cls_layer3 = ConvGNAct(1024, cls_out_dim, k=1, p=0, gn_groups=gn_groups)
        self.cls_layer4 = ConvGNAct(2048, cls_out_dim, k=1, p=0, gn_groups=gn_groups)
        self.cls_fusion = nn.Sequential(
            ConvGNAct(cls_out_dim * 3, cls_out_dim, k=1, p=0, gn_groups=gn_groups),
            ConvGNAct(cls_out_dim, cls_out_dim, k=3, gn_groups=gn_groups),
        )
        self.register_buffer(
            "input_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "input_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    @staticmethod
    def _resize(feature: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        if feature.shape[-2:] == size:
            return feature
        return F.interpolate(feature, size=size, mode="bilinear", align_corners=False)

    def forward(
        self, x: torch.Tensor, return_dict: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Dict[str, object],
    ]:
        if x.dim() != 4 or x.shape[1] not in {1, 3}:
            raise ValueError(f"Expected [B,1|3,H,W], got {tuple(x.shape)}")
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        height, width = x.shape[-2:]
        x = (x - self.input_mean.to(x)) / self.input_std.to(x)

        backbone = self.image_encoder
        stem = backbone.relu(backbone.bn1(backbone.conv1(x)))
        layer1 = backbone.layer1(backbone.maxpool(stem))
        layer2 = backbone.layer2(layer1)
        layer3 = backbone.layer3(layer2)
        layer4 = backbone.layer4(layer3)

        sizes = (
            (height, width),
            (max(1, height // 2), max(1, width // 2)),
            (max(1, height // 4), max(1, width // 4)),
            (max(1, height // 8), max(1, width // 8)),
        )
        seg_feats = (
            self._resize(self.f1_adapter(stem), sizes[0]),
            self._resize(self.f2_adapter(layer1), sizes[1]),
            self._resize(self.f3_adapter(layer2), sizes[2]),
            self._resize(self.f4_adapter(layer4), sizes[3]),
        )

        cls_size = sizes[2]
        cls_feat = self.cls_fusion(torch.cat([
            self._resize(self.cls_layer2(layer2), cls_size),
            self._resize(self.cls_layer3(layer3), cls_size),
            self._resize(self.cls_layer4(layer4), cls_size),
        ], dim=1))

        if not return_dict:
            return seg_feats
        return {
            "seg_feats": seg_feats,
            "cls_feat": cls_feat,
            "intermediate": {
                "layer1": layer1, "layer2": layer2,
                "layer3": layer3, "layer4": layer4,
            },
        }
