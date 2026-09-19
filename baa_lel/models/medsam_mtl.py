"""MedSAM-MTL: a deliberately plain shared-encoder multi-task baseline."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from baa_lel.models.backbones.medsam import MedSAMMultiScaleEncoder
from baa_lel.models.heads import EnhancedClassifier, LesionSegmentationDecoder, build_medsam_fusion_projections


class MedSAMMTLModel(nn.Module):
    """Joint segmentation/classification using only a shared MedSAM encoder.

    No segmentation prediction or morphology/graph representation is consumed
    by the classification branch.  The two tasks interact exclusively through
    the parameters of ``self.encoder``.
    """

    def __init__(
        self,
        clinical_dim: int,
        num_classes: int = 3,
        medsam_checkpoint_path=None,
        medsam_model_type: str = "vit_b",
        freeze_medsam: bool = True,
        unfreeze_medsam_last_n: int = 0,
        medsam_activation_checkpointing: bool = False,
        image_size: int = 1024,
        hidden_dim: int = 256,
        cls_dropout: float = 0.3,
        gn_groups: int = 32,
        **_: object,
    ):
        super().__init__()
        if clinical_dim < 1:
            raise ValueError("MedSAM-MTL requires at least one clinical feature")
        self.clinical_dim = int(clinical_dim)
        self.num_classes = int(num_classes)
        self.lambda_anchor = 0.0

        self.encoder = MedSAMMultiScaleEncoder(
            checkpoint_path=medsam_checkpoint_path,
            model_type=medsam_model_type,
            freeze_medsam=freeze_medsam,
            unfreeze_last_n=unfreeze_medsam_last_n,
            activation_checkpointing=medsam_activation_checkpointing,
            out_channels=(120, 240, 480, 960),
            gn_groups=gn_groups,
            force_img_size=image_size,
            out_indices=(4, 7, 10),
            return_cls_feat=True,
            cls_out_dim=256,
        )


        self.decoder = LesionSegmentationDecoder(gn_groups=gn_groups)
        self.seg_head = nn.Conv2d(120, 1, kernel_size=1, bias=True)


        self.image_projection, self.clinical_projection = build_medsam_fusion_projections(
            self.clinical_dim, hidden_dim
        )
        self.cls_head = EnhancedClassifier(
            feature_dim=hidden_dim * 2,
            output_size=self.num_classes,
            hidden_dims=(hidden_dim,),
            dropout_rate=cls_dropout,
        )


        self.log_var_seg = nn.Parameter(torch.tensor(0.0))
        self.log_var_cls = nn.Parameter(torch.tensor(0.0))
        self._init_task_weights()

    def forward(self, x_img, c_obs=None, m=None, task="both"):
        if task not in {"seg", "cls", "both"}:
            raise ValueError(f"unknown task: {task!r}")
        if x_img.dim() != 4:
            raise ValueError(f"x_img must be [B,C,H,W], got {tuple(x_img.shape)}")
        batch, _, height, width = x_img.shape
        need_seg, need_cls = task in {"seg", "both"}, task in {"cls", "both"}

        encoded = self.encoder(x_img, return_dict=True)
        cls_feat = encoded["cls_feat"]

        feat_seg = seg_logits_low = seg_logits = None
        if need_seg:
            f1, f2, f3, f4 = encoded["seg_feats"]
            feat_seg = self.decoder(f1, f2, f3, f4)
            seg_logits_low = self.seg_head(feat_seg)
            seg_logits = F.interpolate(
                seg_logits_low, size=(height, width), mode="bilinear", align_corners=False
            )

        cls_logits = None
        image_vector = clinical_vector = fused_vector = None
        if need_cls:
            if c_obs is None or c_obs.dim() != 2 or c_obs.shape != (batch, self.clinical_dim):
                shape = None if c_obs is None else tuple(c_obs.shape)
                raise ValueError(f"c_obs must be [B,{self.clinical_dim}], got {shape}")
            if m is not None and m.shape != c_obs.shape:
                raise ValueError(f"m must match c_obs shape, got {tuple(m.shape)}")
            image_vector = self.image_projection(F.adaptive_avg_pool2d(cls_feat, 1).flatten(1))
            clinical_vector = self.clinical_projection(c_obs.float())
            fused_vector = torch.cat([image_vector, clinical_vector], dim=1)
            cls_logits = self.cls_head(fused_vector)

        aux = {
            "seg_logits_low": seg_logits_low,
            "seg_logits_low_raw": seg_logits_low,
            "feat_seg": feat_seg,
            "cls_feat": cls_feat,
            "image_vector": image_vector,
            "clinical_vector": clinical_vector,
            "fused_vector": fused_vector,
        }
        return seg_logits, cls_logits, aux

    def get_total_loss(self, Lseg, Lcls, Lanchor=0.0, lambda_anchor=0.0, clamp=(-5.0, 5.0), **_):
        if float(lambda_anchor) != 0.0:
            raise ValueError("MedSAM-MTL requires lambda_anchor=0")
        lo, hi = clamp
        lv_seg = self.log_var_seg.clamp(lo, hi)
        lv_cls = self.log_var_cls.clamp(lo, hi)
        loss = 0.5 * torch.exp(-lv_seg) * Lseg + 0.5 * lv_seg + 0.5 * torch.exp(-lv_cls) * Lcls + 0.5 * lv_cls
        weights = {
            "w_seg": float((0.5 * torch.exp(-lv_seg)).detach().cpu()),
            "w_cls": float((0.5 * torch.exp(-lv_cls)).detach().cpu()),
            "lambda_anchor": 0.0,
            "Lanchor": 0.0,
        }
        return loss, weights

    def _init_task_weights(self):
        for name, module in self.named_modules():
            if name == "encoder.image_encoder" or name.startswith("encoder.image_encoder."):
                continue
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
