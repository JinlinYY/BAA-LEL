from __future__ import annotations

from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from baa_lel.models.baa_lel import BAALEL


class DiceLossPerSample(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2 * inter + self.eps) / (union + self.eps)
        return (1.0 - dice).view(dice.size(0))


def _soft_boundary(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    padding = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-mask, kernel_size, stride=1, padding=padding)
    return (dilated - eroded).clamp(0.0, 1.0)


class BoundaryDiceLossPerSample(nn.Module):
    """Dice loss on differentiable morphological boundaries."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predicted = _soft_boundary(torch.sigmoid(logits))
        expected = _soft_boundary(targets.float())
        intersection = (predicted * expected).flatten(1).sum(1)
        denominator = predicted.flatten(1).sum(1) + expected.flatten(1).sum(1)
        return 1.0 - (2.0 * intersection + self.eps) / (denominator + self.eps)


def segmentation_loss_per_sample(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    coarse_logits: torch.Tensor | None,
    bce_weight: float,
    dice_weight: float | None,
    boundary_weight: float = 0.0,
    coarse_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the configured four-part segmentation objective per sample."""
    resolved_dice_weight = (
        1.0 - float(bce_weight) if dice_weight is None else float(dice_weight)
    )
    weights = (
        float(bce_weight), resolved_dice_weight,
        float(boundary_weight), float(coarse_weight),
    )
    if any(weight < 0.0 for weight in weights) or abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("segmentation loss weights must be non-negative and sum to 1")

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    bce_per = bce.flatten(1).mean(1)
    dice_per = DiceLossPerSample()(logits, targets)
    boundary_per = BoundaryDiceLossPerSample()(logits, targets)
    if coarse_logits is None:
        if coarse_weight > 0.0:
            raise ValueError("coarse segmentation weight requires seg_logits_low_raw")
        coarse_per = torch.zeros_like(dice_per)
    else:
        coarse_targets = F.interpolate(
            targets.float(), size=coarse_logits.shape[-2:], mode="nearest"
        )
        coarse_per = DiceLossPerSample()(coarse_logits, coarse_targets)

    total = (
        weights[0] * bce_per
        + weights[1] * dice_per
        + weights[2] * boundary_per
        + weights[3] * coarse_per
    )
    return total, {
        "bce": bce_per,
        "dice": dice_per,
        "boundary": boundary_per,
        "coarse": coarse_per,
    }


def make_nomissing_loss_fn(
    cls_criterion: nn.Module,
    seg_bce_weight: float,
    force_full_observed_mask: bool,
    seg_dice_weight: float | None = None,
    seg_boundary_weight: float = 0.0,
    seg_coarse_weight: float = 0.0,
    use_uncertainty_weighting: bool = True,
    lambda_anchor: float = 0.01,
    classification_only: bool = False,
    segmentation_only: bool = False,
) -> Callable:
    """Make nomissing loss fn."""
    def _loss_fn(
        model: BAALEL,
        x_img: torch.Tensor,
        seg_gt: torch.Tensor,
        has_mask: torch.Tensor,
        y_gt: torch.Tensor,
        c_obs: torch.Tensor,
        m: torch.Tensor,
        **kwargs,
    ):
        if force_full_observed_mask:
            m = torch.ones_like(m)


        seg_logits, cls_logits, aux = model(
            x_img,
            c_obs=c_obs,
            m=m,
            task="cls" if classification_only else "both",
        )

        if classification_only:
            Lcls = cls_logits.sum() * 0.0 if segmentation_only else cls_criterion(cls_logits, y_gt)
            return Lcls, {
                "Lseg": 0.0,
                "Lseg_bce": 0.0,
                "Lseg_dice": 0.0,
                "Lseg_boundary": 0.0,
                "Lseg_coarse": 0.0,
                "Lcls": float(Lcls.detach().cpu()),
                "Limp": 0.0,
                "Lcons": 0.0,
                "Lanchor": 0.0,
                "Ltotal": float(Lcls.detach().cpu()),
                "w_seg": 0.0,
                "w_cls": 1.0,
                "w_imp": 0.0,
                "lambda_cons": 0.0,
                "lambda_anchor": 0.0,
            }


        seg_per, seg_components = segmentation_loss_per_sample(
            seg_logits,
            seg_gt,
            coarse_logits=aux.get("seg_logits_low_raw"),
            bce_weight=seg_bce_weight,
            dice_weight=seg_dice_weight,
            boundary_weight=seg_boundary_weight,
            coarse_weight=seg_coarse_weight,
        )

        Lseg = (
            (seg_per * has_mask).sum()
            / has_mask.sum().clamp_min(1.0)
        )


        Lcls = cls_logits.sum() * 0.0 if segmentation_only else cls_criterion(cls_logits, y_gt)


        Lanchor = aux.get("boundary_anchor_loss", None)

        if Lanchor is None:
            Lanchor = torch.zeros(
                (),
                device=x_img.device,
                dtype=Lseg.dtype,
            )


        if use_uncertainty_weighting and not segmentation_only and hasattr(model, "get_total_loss"):
            Ltotal, weights = model.get_total_loss(
                Lseg=Lseg,
                Lcls=Lcls,
                Limp=0.0,
                Lcons=0.0,
                Lanchor=Lanchor,
                lambda_anchor=lambda_anchor,
            )
        else:
            Ltotal = Lseg + Lcls + float(lambda_anchor) * Lanchor
            weights = {
                "w_seg": 1.0,
                "w_cls": 1.0,
                "w_imp": 0.0,
                "lambda_cons": 0.0,
                "lambda_anchor": float(lambda_anchor),
            }


        log = {
            "Lseg": float(Lseg.detach().cpu()),
            "Lseg_bce": float(
                ((seg_components["bce"] * has_mask).sum() / has_mask.sum().clamp_min(1.0)).detach().cpu()
            ),
            "Lseg_dice": float(
                ((seg_components["dice"] * has_mask).sum() / has_mask.sum().clamp_min(1.0)).detach().cpu()
            ),
            "Lseg_boundary": float(
                ((seg_components["boundary"] * has_mask).sum() / has_mask.sum().clamp_min(1.0)).detach().cpu()
            ),
            "Lseg_coarse": float(
                ((seg_components["coarse"] * has_mask).sum() / has_mask.sum().clamp_min(1.0)).detach().cpu()
            ),
            "Lcls": float(Lcls.detach().cpu()),
            "Limp": 0.0,
            "Lcons": 0.0,
            "Lanchor": float(Lanchor.detach().cpu()),
            "Ltotal": float(Ltotal.detach().cpu()),

            "w_seg": float(weights.get("w_seg", 1.0)),
            "w_cls": float(weights.get("w_cls", 1.0)),
            "w_imp": float(weights.get("w_imp", 0.0)),
            "lambda_cons": float(weights.get("lambda_cons", 0.0)),
            "lambda_anchor": float(weights.get("lambda_anchor", lambda_anchor)),
        }

        return Ltotal, log

    return _loss_fn
