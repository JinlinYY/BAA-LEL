from typing import Dict, Callable, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix

from torch.nn import Module
from bua_lel.utils.meters import AverageMeter
from bua_lel.utils.metrics_seg import (
    dice_score_from_logits,
    iou_score_from_logits,
    seg_precision_recall_from_logits,
)
from bua_lel.utils.metrics_cls import compute_classwise_sens_spec
from bua_lel.engine.losses import segmentation_loss_per_sample
from baselines.metrics import segmentation_case_metrics


def _unpack_batch(batch):
    if len(batch) == 8:
        img, seg_gt, has_mask, c_obs, m, y, cat_targets, pids = batch
    elif len(batch) == 7:
        img, seg_gt, has_mask, c_obs, m, y, cat_targets = batch
        pids = None
    else:
        raise ValueError(f"Expected a 7- or 8-item batch, got {len(batch)}")
    return img, seg_gt, has_mask, c_obs, m, y, cat_targets, pids

@torch.no_grad()
def evaluate(
    model: Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    seg_thr: float,
    cls_criterion: nn.Module,
    seg_bce_weight: float,
    force_full_observed_mask: bool,
    return_preds: bool = False,
    compute_surface_metrics: bool = False,
    return_case_records: bool = False,
    classification_only: bool = False,
    segmentation_only: bool = False,
    seg_dice_weight: Optional[float] = None,
    seg_boundary_weight: float = 0.0,
    seg_coarse_weight: float = 0.0,
) -> Dict[str, float]:
    model.eval()

    loss_meter = AverageMeter()
    seg_loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()

    dice_meter = AverageMeter()
    miou_meter = AverageMeter()
    segP_meter = AverageMeter()
    segR_meter = AverageMeter()
    hd95_values, assd_values = [], []

    y_true_all, y_pred_all, y_proba_all = [], [], []
    case_records = []

    for batch in loader:
        img, seg_gt, has_mask, c_obs, m, y, cat_targets, pids = _unpack_batch(batch)
        if return_case_records and pids is None:
            raise ValueError("return_case_records=True requires dataset return_pid=True")

        img = img.to(device, non_blocking=True)
        seg_gt = seg_gt.to(device, non_blocking=True)

        has_mask = has_mask.to(device, non_blocking=True).view(-1).float()
        c_obs = c_obs.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if force_full_observed_mask:
            m = torch.ones_like(m)

        seg_logits, cls_logits, aux = model(
            img, c_obs=c_obs, m=m, task="cls" if classification_only else "both"
        )

        if classification_only:
            Lseg = cls_logits.new_zeros(())
        else:
            seg_per, _ = segmentation_loss_per_sample(
                seg_logits,
                seg_gt,
                coarse_logits=aux.get("seg_logits_low_raw"),
                bce_weight=seg_bce_weight,
                dice_weight=seg_dice_weight,
                boundary_weight=seg_boundary_weight,
                coarse_weight=seg_coarse_weight,
            )
            Lseg = (seg_per * has_mask).sum() / has_mask.sum().clamp_min(1.0)

        Lcls = cls_logits.sum() * 0.0 if segmentation_only else cls_criterion(cls_logits, y)
        L = Lseg + Lcls

        loss_meter.update(L.item(), img.size(0))
        seg_loss_meter.update(Lseg.item(), img.size(0))
        cls_loss_meter.update(Lcls.item(), img.size(0))

        if not classification_only and has_mask.sum().item() > 0:
            keep = has_mask.bool()
            dice_meter.update(dice_score_from_logits(seg_logits[keep], seg_gt[keep], thr=seg_thr), int(keep.sum().item()))
            miou_meter.update(iou_score_from_logits(seg_logits[keep], seg_gt[keep], thr=seg_thr), int(keep.sum().item()))
            p, r = seg_precision_recall_from_logits(seg_logits[keep], seg_gt[keep], thr=seg_thr)
            segP_meter.update(p, int(keep.sum().item()))
            segR_meter.update(r, int(keep.sum().item()))
            if compute_surface_metrics:
                pred_masks = (torch.sigmoid(seg_logits[keep]) > seg_thr).cpu().numpy()
                true_masks = (seg_gt[keep] > 0.5).cpu().numpy()
                for pred_mask, true_mask in zip(pred_masks[:, 0], true_masks[:, 0]):
                    surface = segmentation_case_metrics(pred_mask, true_mask)
                    hd95_values.append(surface["HD95"])
                    assd_values.append(surface["ASSD"])

        prob = torch.softmax(cls_logits, dim=1).detach().cpu().numpy()
        pred = np.argmax(prob, axis=1)
        y_true_all.append(y.detach().cpu().numpy())
        y_pred_all.append(pred)
        y_proba_all.append(prob)
        if return_case_records:
            pred_masks_all = (
                None if classification_only
                else (torch.sigmoid(seg_logits) > seg_thr).cpu().numpy()[:, 0]
            )
            true_masks_all = (
                None if classification_only else (seg_gt > 0.5).cpu().numpy()[:, 0]
            )
            has_mask_cpu = has_mask.bool().cpu().numpy()
            y_cpu = y.detach().cpu().numpy()
            for i, pid in enumerate(pids):
                row = {
                    "pid": str(pid), "y_true": int(y_cpu[i]), "y_pred": int(pred[i]),
                    **{f"prob_c{c}": float(prob[i, c]) for c in range(num_classes)},
                    "has_mask": int(has_mask_cpu[i]),
                }
                if has_mask_cpu[i] and not classification_only:
                    row.update(segmentation_case_metrics(pred_masks_all[i], true_masks_all[i]))
                else:
                    row.update({k: float("nan") for k in ("Dice", "mIoU", "HD95", "ASSD")})
                row["segmentation_applicable"] = not classification_only
                case_records.append(row)

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    y_proba = np.concatenate(y_proba_all, axis=0)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    try:
        if num_classes == 2:
            auc_macro_ovr = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc_macro_ovr = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
    except Exception:
        auc_macro_ovr = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    sens, spec = compute_classwise_sens_spec(cm)
    sens_macro = float(np.nanmean(sens))
    spec_macro = float(np.nanmean(spec))

    out = {
        "total_loss": loss_meter.avg,
        "seg_loss": seg_loss_meter.avg,
        "cls_loss": cls_loss_meter.avg,
        "dice": float("nan") if classification_only else dice_meter.avg,
        "miou": float("nan") if classification_only else miou_meter.avg,
        "seg_precision": float("nan") if classification_only else segP_meter.avg,
        "seg_recall": float("nan") if classification_only else segR_meter.avg,
        "hd95": float(np.mean(hd95_values)) if hd95_values else float("nan"),
        "assd": float(np.mean(assd_values)) if assd_values else float("nan"),
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "auc_macro_ovr": float(auc_macro_ovr),
        "sens_macro": sens_macro,
        "spec_macro": spec_macro,
    }
    for c in range(num_classes):
        out[f"sens_c{c}"] = float(sens[c])
        out[f"spec_c{c}"] = float(spec[c])

    if return_preds:
        out["_y_true"] = y_true
        out["_y_proba"] = y_proba
    if return_case_records:
        out["_case_records"] = case_records

    return out

def train_one_epoch(
    model: Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler_amp: torch.cuda.amp.GradScaler,
    loss_fn: Callable,
    max_grad_norm: float,
    grad_accum_steps: int = 1,
) -> Dict[str, float]:
    model.train()

    total_meter = AverageMeter()
    seg_meter = AverageMeter()
    cls_meter = AverageMeter()
    seg_component_meters = {
        key: AverageMeter()
        for key in ("Lseg_bce", "Lseg_dice", "Lseg_boundary", "Lseg_coarse")
    }

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        img, seg_gt, has_mask, c_obs, m, y, cat_targets, _ = _unpack_batch(batch)

        img = img.to(device, non_blocking=True)
        seg_gt = seg_gt.to(device, non_blocking=True)

        has_mask = has_mask.to(device, non_blocking=True).view(-1).float()
        c_obs = c_obs.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            Ltotal, log = loss_fn(
                model=model,
                x_img=img,
                seg_gt=seg_gt,
                has_mask=has_mask,
                y_gt=y,
                c_obs=c_obs,
                m=m,
            )
            Ltotal_scaled = Ltotal / max(1, grad_accum_steps)

        scaler_amp.scale(Ltotal_scaled).backward()

        bs = img.size(0)
        total_meter.update(float(Ltotal.detach().cpu()), bs)
        seg_meter.update(float(log.get("Lseg", 0.0)), bs)
        cls_meter.update(float(log.get("Lcls", 0.0)), bs)
        for key, meter in seg_component_meters.items():
            meter.update(float(log.get(key, 0.0)), bs)

        if (step % grad_accum_steps) == 0:
            scaler_amp.unscale_(optimizer)
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            optimizer.zero_grad(set_to_none=True)

    if (len(loader) % max(1, grad_accum_steps)) != 0:
        scaler_amp.unscale_(optimizer)
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler_amp.step(optimizer)
        scaler_amp.update()
        optimizer.zero_grad(set_to_none=True)

    return {
        "total_loss": total_meter.avg,
        "seg_loss": seg_meter.avg,
        "cls_loss": cls_meter.avg,
        "imp_loss": 0.0,
        "cons_loss": 0.0,
        "seg_bce_loss": seg_component_meters["Lseg_bce"].avg,
        "seg_dice_loss": seg_component_meters["Lseg_dice"].avg,
        "seg_boundary_loss": seg_component_meters["Lseg_boundary"].avg,
        "seg_coarse_loss": seg_component_meters["Lseg_coarse"].avg,
    }
