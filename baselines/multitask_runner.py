"""Nested five-fold runner for genuine multi-task methods."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baselines.data import BaselineDataset, CaseRecord, ClinicalPreprocessor, load_clinical_frame
from baselines.metrics import classification_metrics, segmentation_case_metrics
from baselines.models import MODEL_METADATA, build_model
from baselines.protocol import (
    PROTOCOL_VERSION,
    class_names,
    create_or_load_inner_manifest,
    create_or_load_patient_manifest,
    protocol_records,
    records_for_pids,
    validate_shared_segmentation_folds,
)
from baselines.results import normalize_oof_frame, save_per_class_metrics
from baselines.runner import _dice_loss, seed_everything
from baa_lel.data.paired_transforms import build_training_transform
from baa_lel.engine.losses import segmentation_loss_per_sample


@dataclass
class MultitaskConfig:
    dataset: str
    method: str
    data_root: str = "data"
    output_root: str = "outputs/baselines"
    split_root: str | None = None
    seed: int = 42
    folds: int = 5
    image_size: int = 256
    medsam_input_size: int = 1024
    epochs: int = 50
    patience: int = 12
    batch_size: int = 4
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    threshold: float = .5
    medsam_checkpoint: str = "checkpoints/medsam_vit_b.pth"
    inner_validation_fraction: float = .2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    resume: bool = False
    late_raw_clinical_fusion: bool = False
    unfreeze_medsam_last_n: int = 0
    lambda_anchor: float = 0.0
    architecture_profile: str = "extended"
    clinical_filename: str = "clinical.xlsx"
    fixed_clinical_category_schema: bool = False
    clinical_category_schema: dict[str, list[str]] | None = None
    classifier_fused_dim: int = 256
    selected_folds: tuple[int, ...] | None = None
    keep_fold_checkpoints: bool = True
    train_augmentation: bool = False
    train_augmentation_profile: str = "none"
    seg_bce_weight: float = 0.5
    seg_dice_weight: float | None = None
    seg_boundary_weight: float = 0.0
    seg_coarse_weight: float = 0.0
    cls_lr_mult: float = 1.0
    selection_weight_dice: float = .3
    selection_weight_macro_f1: float = .7
    clinical_exclude_columns: tuple[str, ...] = ()
    modality: str = "multimodal"
    boundary_radius_ratio: float = 0.15
    peritumor_radius_ratio: float = 0.40
    min_boundary_radius: int = 2
    max_boundary_radius: int = 4
    min_peritumor_radius: int = 4
    max_peritumor_radius: int = 8


def _dataset(records: Sequence[CaseRecord], clinical: np.ndarray, cfg: MultitaskConfig, *, augment=False):
    transform = (
        build_training_transform(cfg.train_augmentation_profile)
        if augment and cfg.train_augmentation_profile != "none"
        else None
    )
    return BaselineDataset(
        records,
        clinical,
        cfg.image_size,
        augment=bool(augment and cfg.train_augmentation_profile == "none"),
        grayscale=cfg.dataset.lower() != "imaplusplus",
        transform=transform,
    )


def _loader(dataset, cfg: MultitaskConfig, shuffle: bool):
    worker_args = {"persistent_workers": True, "prefetch_factor": 8} if cfg.num_workers > 0 else {}
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers, pin_memory=(torch.device(cfg.device).type == "cuda"), **worker_args)


def _class_criterion(records, num_classes, device):
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1)
    weights /= weights.mean()
    return nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float32, device=device), label_smoothing=.05)


def _clinical_feature_layout(preprocessor: ClinicalPreprocessor):
    numeric_count = len(preprocessor.numeric_columns)
    categorical_slices = preprocessor.feature_slices_[numeric_count:]
    onehot_slices = {
        str(column): tuple(feature_slice)
        for column, feature_slice in zip(
            preprocessor.categorical_columns, categorical_slices
        )
    }
    return (0, numeric_count), onehot_slices


def _fingerprint_payload(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _clinical_workbook_path(
    dataset: str, data_root: str | Path, clinical_filename: str | Path = "clinical.xlsx"
) -> Path | None:
    folder = {
        "her2usc": "HER2",
        "lmnusc": "LNM",
        "breast": "BrEaST",
    }.get(dataset.lower())
    return Path(data_root) / folder / clinical_filename if folder else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_asset_fingerprint(records: Sequence[CaseRecord]) -> str:
    """Fingerprint every image/mask byte stream paired to a labelled case."""
    payload = []
    for record in sorted(records, key=lambda item: str(item.pid)):
        paths = (record.image_path, *record.mask_paths)
        payload.append({
            "pid": str(record.pid),
            "label": int(record.label),
            "image": {
                "path": str(record.image_path.resolve()),
                "sha256": _file_sha256(record.image_path) if record.image_path.is_file() else None,
            },
            "masks": [
                {
                    "path": str(Path(path).resolve()),
                    "sha256": _file_sha256(Path(path)) if Path(path).is_file() else None,
                }
                for path in record.mask_paths
            ],
        })
    return _fingerprint_payload(payload)


def _configured_clinical_categories(
    cfg: MultitaskConfig,
    clinical: pd.DataFrame,
) -> dict[str, list[str]] | None:
    if not cfg.fixed_clinical_category_schema:
        return None
    if not cfg.clinical_category_schema:
        raise ValueError(
            "fixed clinical category schema requires a predeclared category dictionary"
        )
    configured = {
        str(column): list(map(str, values))
        for column, values in cfg.clinical_category_schema.items()
    }
    numeric = set(clinical.select_dtypes(include=[np.number]).columns)
    categorical = {str(column) for column in clinical.columns if column not in numeric}
    if set(configured) != categorical:
        raise ValueError(
            "clinical category schema keys do not exactly match categorical columns: "
            f"expected {sorted(categorical)}, got {sorted(configured)}"
        )
    return configured


def _preprocessor_provenance(
    frame: pd.DataFrame,
    fixed_categories: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    preprocessor = ClinicalPreprocessor(fixed_categories=fixed_categories).fit(frame)
    state = preprocessor.state_dict()
    feature_dim = int(preprocessor.transform(frame.iloc[:0]).shape[1])
    return {
        "clinical_feature_dim": feature_dim,
        "preprocessor_sha256": _fingerprint_payload(state),
        "preprocessor_state": state,
    }


def build_run_provenance(
    cfg: MultitaskConfig,
    records: Sequence[CaseRecord],
    clinical: pd.DataFrame,
    outer: dict[str, object],
    inner: dict[str, object],
) -> dict[str, object]:
    """Build the exact data/protocol signature required to reuse fold outputs."""
    workbook = _clinical_workbook_path(
        cfg.dataset, cfg.data_root, cfg.clinical_filename
    )
    fixed_categories = _configured_clinical_categories(cfg, clinical)
    inner_by_fold = {int(item["fold"]): item for item in inner["folds"]}
    fold_metadata: dict[str, object] = {}
    for outer_fold in outer["folds"]:
        fold = int(outer_fold["fold"])
        inner_fold = inner_by_fold[fold]
        selection_pids = list(map(str, inner_fold["train_pids"]))
        final_pids = list(map(str, outer_fold["train_pids"]))
        fold_metadata[str(fold)] = {
            "selection": _preprocessor_provenance(
                clinical.loc[selection_pids], fixed_categories
            ),
            "final": _preprocessor_provenance(clinical.loc[final_pids], fixed_categories),
        }
    record_labels = sorted(
        ({"pid": str(record.pid), "label": int(record.label)} for record in records),
        key=lambda item: item["pid"],
    )
    workbook_metadata = {
        "path": str(workbook.resolve()) if workbook is not None else None,
        "sha256": _file_sha256(workbook) if workbook is not None and workbook.is_file() else None,
    }
    return {
        "schema_version": 1,
        "dataset": cfg.dataset,
        "method": cfg.method,
        "clinical_workbook": workbook_metadata,
        "clinical_columns": [str(column) for column in clinical.columns],
        "clinical_raw_dim": int(clinical.shape[1]),
        "patient_count": len(records),
        "patient_label_fingerprint": _fingerprint_payload(record_labels),
        "case_asset_fingerprint": _case_asset_fingerprint(records),
        "outer_fold_fingerprint": outer["split_fingerprint"],
        "inner_fold_fingerprint": inner["split_fingerprint"],
        "training_protocol": {
            "seed": int(cfg.seed),
            "folds": int(cfg.folds),
            "epochs": int(cfg.epochs),
            "patience": int(cfg.patience),
            "lr": float(cfg.lr),
            "weight_decay": float(cfg.weight_decay),
            "batch_size": int(cfg.batch_size),
            "grad_accum_steps": int(cfg.grad_accum_steps),
            "num_workers": int(cfg.num_workers),
            "inner_validation_fraction": float(cfg.inner_validation_fraction),
            "late_raw_clinical_fusion": bool(cfg.late_raw_clinical_fusion),
            "unfreeze_medsam_last_n": int(cfg.unfreeze_medsam_last_n),
            "lambda_anchor": float(cfg.lambda_anchor),
            "architecture_profile": cfg.architecture_profile,
            "fixed_clinical_category_schema": bool(
                cfg.fixed_clinical_category_schema
            ),
            "clinical_category_schema": fixed_categories,
            "classifier_fused_dim": int(cfg.classifier_fused_dim),
            "modality": cfg.modality,
            "boundary_radius_ratio": float(cfg.boundary_radius_ratio),
            "peritumor_radius_ratio": float(cfg.peritumor_radius_ratio),
            "min_boundary_radius": int(cfg.min_boundary_radius),
            "max_boundary_radius": int(cfg.max_boundary_radius),
            "min_peritumor_radius": int(cfg.min_peritumor_radius),
            "max_peritumor_radius": int(cfg.max_peritumor_radius),
            "selected_folds": list(cfg.selected_folds) if cfg.selected_folds else None,
            "keep_fold_checkpoints": bool(cfg.keep_fold_checkpoints),
            "train_augmentation": bool(cfg.train_augmentation),
            "train_augmentation_profile": cfg.train_augmentation_profile,
            "seg_bce_weight": float(cfg.seg_bce_weight),
            "seg_dice_weight": (
                None if cfg.seg_dice_weight is None else float(cfg.seg_dice_weight)
            ),
            "seg_boundary_weight": float(cfg.seg_boundary_weight),
            "seg_coarse_weight": float(cfg.seg_coarse_weight),
            "cls_lr_mult": float(cfg.cls_lr_mult),
            "selection_weight_dice": float(cfg.selection_weight_dice),
            "selection_weight_macro_f1": float(cfg.selection_weight_macro_f1),
        },
        "folds": fold_metadata,
    }


def validate_or_write_run_provenance(
    root: Path,
    current: dict[str, object],
    *,
    resume: bool,
) -> Path:
    """Prevent resume from mixing outputs produced by different clinical data."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "run_provenance.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if resume and existing != current:
            raise ValueError(
                "run provenance mismatch: clinical data, patient labels, folds, or "
                "training protocol changed; use a fresh output root"
            )
    elif resume and any(candidate.is_file() for candidate in root.glob("fold*/*")):
        raise ValueError(
            "missing run provenance for existing fold artifacts; use a fresh output root"
        )
    path.write_text(
        json.dumps(current, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def _build(
    cfg: MultitaskConfig,
    clinical_dim: int,
    num_classes: int,
    numeric_slice=None,
    onehot_slices_dict=None,
):
    if cfg.method == "mtanet":
        return build_model("mtanet", "multitask", clinical_dim=clinical_dim, num_classes=num_classes, base_channels=32)
    if cfg.method == "medsam_mtl":
        return build_model(
            "medsam_mtl", "multitask", clinical_dim=clinical_dim, num_classes=num_classes,
            medsam_checkpoint_path=cfg.medsam_checkpoint, image_size=cfg.medsam_input_size,
            freeze_medsam=True, unfreeze_medsam_last_n=0,
        )
    if cfg.method != "baa_lel":
        raise ValueError("multitask runner only supports baa_lel, medsam_mtl and mtanet")
    return build_model(
        "baa_lel", "multitask", clinical_dim=clinical_dim,
        numeric_slice=numeric_slice or (0, clinical_dim),
        onehot_slices_dict=onehot_slices_dict or {},
        num_classes=num_classes, medsam_checkpoint_path=cfg.medsam_checkpoint, image_size=cfg.medsam_input_size,
        freeze_medsam=True, unfreeze_medsam_last_n=cfg.unfreeze_medsam_last_n,
        lambda_anchor=cfg.lambda_anchor,
        late_raw_clinical_fusion=cfg.late_raw_clinical_fusion,
        classifier_fused_dim=cfg.classifier_fused_dim,
        architecture_profile=cfg.architecture_profile,
        modality=cfg.modality,
        boundary_radius_ratio=cfg.boundary_radius_ratio,
        peritumor_radius_ratio=cfg.peritumor_radius_ratio,
        min_boundary_radius=cfg.min_boundary_radius,
        max_boundary_radius=cfg.max_boundary_radius,
        min_peritumor_radius=cfg.min_peritumor_radius,
        max_peritumor_radius=cfg.max_peritumor_radius,
    )


def _checkpoint_model_metadata(
    cfg: MultitaskConfig,
    clinical_dim: int,
    num_classes: int,
) -> dict[str, object]:
    """Return enough architecture metadata to rebuild a persisted model."""
    return {
        "method": cfg.method,
        "clinical_feature_dim": int(clinical_dim),
        "num_classes": int(num_classes),
        "architecture_profile": cfg.architecture_profile,
        "late_raw_clinical_fusion": bool(cfg.late_raw_clinical_fusion),
        "classifier_fused_dim": int(cfg.classifier_fused_dim),
        "modality": cfg.modality,
    }


def _forward(model, method, image, clinical):
    if method == "mtanet":
        output = model(image, clinical)
        return output["segmentation_logits"], output["classification_logits"], output.get("coarse_logits")
    seg_logits, cls_logits, aux = model(image, c_obs=clinical, m=torch.ones_like(clinical), task="both")
    return seg_logits, cls_logits, aux


def _loss(
    model,
    method,
    image,
    mask,
    has_mask,
    clinical,
    labels,
    class_criterion,
    cfg,
    *,
    return_components: bool = False,
):
    seg_logits, cls_logits, extra = _forward(model, method, image, clinical)
    modern_segmentation_objective = (
        cfg.seg_bce_weight != 0.5
        or cfg.seg_dice_weight is not None
        or cfg.seg_boundary_weight != 0.0
        or cfg.seg_coarse_weight != 0.0
    )
    if modern_segmentation_objective:
        coarse_logits = (
            extra if method == "mtanet"
            else extra.get("seg_logits_low_raw") if isinstance(extra, dict) else None
        )
        seg_per, seg_components = segmentation_loss_per_sample(
            seg_logits,
            mask,
            coarse_logits=coarse_logits,
            bce_weight=cfg.seg_bce_weight,
            dice_weight=cfg.seg_dice_weight,
            boundary_weight=cfg.seg_boundary_weight,
            coarse_weight=cfg.seg_coarse_weight,
        )
    else:
        bce = nn.BCEWithLogitsLoss(reduction="none")
        bce_per = bce(seg_logits, mask).flatten(1).mean(1)
        probability = torch.sigmoid(seg_logits)
        dice_per = 1.0 - (
            (2 * (probability * mask).flatten(1).sum(1) + 1)
            / (probability.flatten(1).sum(1) + mask.flatten(1).sum(1) + 1)
        )
        seg_per = .5 * bce_per + .5 * dice_per
        seg_components = {
            "bce": bce_per,
            "dice": dice_per,
            "boundary": torch.zeros_like(dice_per),
            "coarse": torch.zeros_like(dice_per),
        }
    segmentation = (seg_per * has_mask).sum() / has_mask.sum().clamp_min(1)
    if not modern_segmentation_objective and extra is not None and method == "mtanet":
        coarse_per = nn.functional.binary_cross_entropy_with_logits(extra, mask, reduction="none").flatten(1).mean(1)
        coarse = (coarse_per * has_mask).sum() / has_mask.sum().clamp_min(1)
        segmentation = segmentation + .25 * coarse
    classification = class_criterion(cls_logits, labels)
    if method in {"baa_lel", "medsam_mtl"} and hasattr(model, "get_total_loss"):
        anchor_loss = (
            extra.get(
                "boundary_anchor_loss",
                extra.get("anchor_loss", segmentation.new_zeros(())),
            )
            if isinstance(extra, dict)
            else segmentation.new_zeros(())
        )
        total, _ = model.get_total_loss(
            segmentation,
            classification,
            Lanchor=anchor_loss,
            lambda_anchor=cfg.lambda_anchor if method == "baa_lel" else 0.0,
        )
    else:
        total = segmentation + classification
    if not return_components:
        return total, segmentation.detach(), classification.detach()
    component_means = {
        f"seg_{name}": float(
            ((values.detach() * has_mask).sum() / has_mask.sum().clamp_min(1)).cpu()
        )
        for name, values in seg_components.items()
    }
    component_means.update({
        "segmentation": float(segmentation.detach().cpu()),
        "classification": float(classification.detach().cpu()),
        "anchor": float(anchor_loss.detach().cpu()) if torch.is_tensor(anchor_loss) else 0.0,
    })
    return total, segmentation.detach(), classification.detach(), component_means


def _optimizer(model: nn.Module, cfg: MultitaskConfig) -> torch.optim.Optimizer:
    """Give the final classifier the requested learning-rate multiplier."""
    if cfg.cls_lr_mult <= 0.0:
        raise ValueError("cls_lr_mult must be positive")
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    classifier = [
        parameter for name, parameter in named
        if name.startswith("cls_head.")
    ]
    classifier_ids = {id(parameter) for parameter in classifier}
    base = [parameter for _, parameter in named if id(parameter) not in classifier_ids]
    groups = []
    if base:
        groups.append({"params": base, "lr": cfg.lr})
    if classifier:
        groups.append({"params": classifier, "lr": cfg.lr * cfg.cls_lr_mult})
    if not groups:
        raise RuntimeError("model has no trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


def _train_epoch(model, loader, optimizer, criterion, cfg, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))
    accumulation, losses = max(1, cfg.grad_accum_steps), []
    component_history = []
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, 1):
        image, mask = batch["image"].to(device), batch["mask"].to(device)
        clinical, labels = batch["clinical"].to(device), batch["label"].to(device)
        has_mask = batch["has_mask"].to(device).view(-1).float()
        with torch.autocast(device_type=device.type, enabled=(cfg.amp and device.type == "cuda")):
            loss, _, _, components = _loss(
                model, cfg.method, image, mask, has_mask, clinical, labels,
                criterion, cfg, return_components=True,
            )
        scaler.scale(loss / accumulation).backward()
        losses.append(float(loss.detach().cpu()))
        component_history.append(components)
        if step % accumulation == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
    history = {"train_loss": float(np.mean(losses))}
    for key in ("segmentation", "classification", "anchor", "seg_bce", "seg_dice", "seg_boundary", "seg_coarse"):
        history[f"train_{key}"] = float(np.mean([item[key] for item in component_history]))
    return history


@torch.no_grad()
def _evaluate(model, loader, cfg, device, fold=None):
    model.eval(); classification_rows, segmentation_rows = [], []
    for batch in loader:
        image, clinical = batch["image"].to(device), batch["clinical"].to(device)
        seg_logits, cls_logits, _ = _forward(model, cfg.method, image, clinical)
        probabilities = torch.softmax(cls_logits, 1).cpu().numpy()
        masks = (torch.sigmoid(seg_logits) >= cfg.threshold).cpu().numpy()
        true_masks, has_masks, labels = batch["mask"].numpy(), batch["has_mask"].numpy().astype(bool), batch["label"].numpy()
        for index, pid in enumerate(batch["pid"]):
            cls_row = {"pid": str(pid), "y_true": int(labels[index]), "y_pred": int(probabilities[index].argmax())}
            cls_row.update({f"proba_{column}": float(value) for column, value in enumerate(probabilities[index])})
            if fold is not None: cls_row["fold"] = int(fold)
            classification_rows.append(cls_row)
            if has_masks[index]:
                seg_row = {"pid": str(pid), **segmentation_case_metrics(masks[index], true_masks[index])}
                if fold is not None: seg_row["fold"] = int(fold)
                segmentation_rows.append(seg_row)
    return classification_rows, segmentation_rows


def _clinical_arrays(frame, preprocessor, groups):
    return [preprocessor.transform(frame.loc[[record.pid for record in group]]) for group in groups]


def _select_epoch(cfg, train_records, dev_records, clinical, names, device, fold_dir, fold, fixed_categories=None):
    preprocessor = ClinicalPreprocessor(fixed_categories=fixed_categories).fit(clinical.loc[[record.pid for record in train_records]])
    train_clinical, dev_clinical = _clinical_arrays(clinical, preprocessor, [train_records, dev_records])
    train_loader, dev_loader = _loader(_dataset(train_records, train_clinical, cfg, augment=cfg.train_augmentation), cfg, True), _loader(_dataset(dev_records, dev_clinical, cfg), cfg, False)
    seed_everything(cfg.seed + fold)
    numeric_slice, onehot_slices = _clinical_feature_layout(preprocessor)
    model = _build(
        cfg, train_clinical.shape[1], len(names), numeric_slice, onehot_slices
    ).to(device)
    optimizer = _optimizer(model, cfg)
    criterion = _class_criterion(train_records, len(names), device)
    best_epoch, best_score, stale, history = 1, -np.inf, 0, []
    for epoch in range(1, cfg.epochs + 1):
        train_history = _train_epoch(model, train_loader, optimizer, criterion, cfg, device)
        cls_rows, seg_rows = _evaluate(model, dev_loader, cfg, device)
        macro_f1 = classification_metrics(pd.DataFrame(cls_rows)["y_true"], pd.DataFrame(cls_rows)[[f"proba_{name}" for name in names]].to_numpy(), names)["Macro-F1"]
        dice = float(pd.DataFrame(seg_rows)["Dice"].mean()) if seg_rows else 0.0
        score = cfg.selection_weight_dice * dice + cfg.selection_weight_macro_f1 * macro_f1
        history.append({
            "fold": fold,
            "epoch": epoch,
            **train_history,
            "development_dice": dice,
            "development_macro_f1": macro_f1,
            "development_score": score,
        })
        if score > best_score: best_epoch, best_score, stale = epoch, score, 0
        else:
            stale += 1
            if stale >= cfg.patience: break
    pd.DataFrame(history).to_csv(fold_dir / "selection_metrics.csv", index=False, encoding="utf-8-sig")
    return best_epoch, best_score


def _final_predict(cfg, train_records, test_records, clinical, names, selected_epoch, device, fold_dir, fold, fixed_categories=None):
    preprocessor = ClinicalPreprocessor(fixed_categories=fixed_categories).fit(clinical.loc[[record.pid for record in train_records]])
    train_clinical, test_clinical = _clinical_arrays(clinical, preprocessor, [train_records, test_records])
    train_loader, test_loader = _loader(_dataset(train_records, train_clinical, cfg, augment=cfg.train_augmentation), cfg, True), _loader(_dataset(test_records, test_clinical, cfg), cfg, False)
    seed_everything(cfg.seed + fold)
    numeric_slice, onehot_slices = _clinical_feature_layout(preprocessor)
    model = _build(
        cfg, train_clinical.shape[1], len(names), numeric_slice, onehot_slices
    ).to(device)
    optimizer = _optimizer(model, cfg)
    criterion = _class_criterion(train_records, len(names), device)
    for _ in range(selected_epoch): _train_epoch(model, train_loader, optimizer, criterion, cfg, device)
    checkpoint_path = fold_dir / ("best_checkpoint.pt" if cfg.method == "medsam_mtl" else "model.pt")
    model_metadata = _checkpoint_model_metadata(
        cfg, train_clinical.shape[1], len(names)
    )
    torch.save({
        "model_state": model.state_dict(),
        "selected_epoch": selected_epoch,
        "fold": fold,
        "method": cfg.method,
        "clinical_feature_dim": int(train_clinical.shape[1]),
        "classifier_fused_dim": int(cfg.classifier_fused_dim),
        "late_raw_clinical_fusion": bool(cfg.late_raw_clinical_fusion),
        "architecture_profile": cfg.architecture_profile,
        "modality": cfg.modality,
        "num_classes": len(names),
        "model_config": model_metadata,
        "clinical_preprocessor": _preprocessor_provenance(
            clinical.loc[[record.pid for record in train_records]], fixed_categories
        ),
    }, checkpoint_path)
    if cfg.method == "medsam_mtl":

        saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(saved["model_state"], strict=True)
    predictions = _evaluate(model, test_loader, cfg, device, fold)
    if not cfg.keep_fold_checkpoints:
        checkpoint_path.unlink()
    return predictions


def validate_multitask_oof(
    classification: pd.DataFrame,
    segmentation: pd.DataFrame,
    records: Sequence[CaseRecord],
) -> None:
    """Validate exact patient coverage and finite values before publishing OOF."""
    if classification["pid"].duplicated().any() or segmentation["pid"].duplicated().any():
        raise RuntimeError("duplicate multi-task OOF patients")
    if set(classification["pid"].astype(str)) != {record.pid for record in records}:
        raise RuntimeError("classification OOF coverage mismatch")
    if set(segmentation["pid"].astype(str)) != {
        record.pid for record in records if record.mask_paths
    }:
        raise RuntimeError("segmentation OOF coverage mismatch")
    for task, frame in (("classification", classification), ("segmentation", segmentation)):
        numeric = frame.select_dtypes(include=[np.number])
        if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{task} OOF contains non-finite values")


def run_multitask_benchmark(cfg: MultitaskConfig):
    if cfg.method not in {"baa_lel", "medsam_mtl", "mtanet"}: raise ValueError("only BAA-LEL, MedSAM-MTL and MTANet are valid multi-task methods")
    if cfg.train_augmentation_profile not in {
        "none", "ultrasound_mild", "ultrasound_gentle", "dermoscopy_mild",
    }:
        raise ValueError("unknown train_augmentation_profile")
    resolved_dice_weight = 1.0 - float(cfg.seg_bce_weight) if cfg.seg_dice_weight is None else float(cfg.seg_dice_weight)
    segmentation_weights = (
        float(cfg.seg_bce_weight), resolved_dice_weight,
        float(cfg.seg_boundary_weight), float(cfg.seg_coarse_weight),
    )
    if any(weight < 0.0 for weight in segmentation_weights) or not np.isclose(sum(segmentation_weights), 1.0):
        raise ValueError("segmentation loss weights must be non-negative and sum to 1")
    if cfg.modality not in {"multimodal", "ultrasound_only", "clinical_only"}:
        raise ValueError("unknown modality")
    if cfg.method != "baa_lel" and cfg.modality != "multimodal":
        raise ValueError("single-modality settings are supported only by baa_lel")
    device = torch.device(cfg.device)
    records = protocol_records(
        cfg.dataset, cfg.data_root, clinical_name=cfg.clinical_filename
    )
    split_root = cfg.split_root or cfg.output_root
    outer = create_or_load_patient_manifest(split_root, cfg.dataset, records, folds=cfg.folds, seed=cfg.seed)
    inner = create_or_load_inner_manifest(split_root, cfg.dataset, records, outer, seed=cfg.seed, validation_fraction=cfg.inner_validation_fraction)
    validate_shared_segmentation_folds(outer, records)
    clinical = load_clinical_frame(
        cfg.dataset,
        Path(cfg.data_root),
        records,
        clinical_name=cfg.clinical_filename,
    )
    missing_exclusions = [
        column for column in cfg.clinical_exclude_columns if column not in clinical.columns
    ]
    if missing_exclusions:
        raise ValueError(f"clinical exclusion columns not found: {missing_exclusions}")
    clinical = clinical.drop(columns=list(cfg.clinical_exclude_columns))
    names = class_names(records)
    fixed_categories = _configured_clinical_categories(cfg, clinical)
    root = Path(cfg.output_root) / cfg.dataset / "multitask" / cfg.method
    root.mkdir(parents=True, exist_ok=True)
    provenance = build_run_provenance(cfg, records, clinical, outer, inner)
    validate_or_write_run_provenance(root, provenance, resume=cfg.resume)
    inner_by_fold = {int(item["fold"]): item for item in inner["folds"]}
    all_cls, all_seg, fold_rows = [], [], []
    selected = set(cfg.selected_folds or range(1, cfg.folds + 1))
    for outer_fold in outer["folds"]:
        fold, fold_dir = int(outer_fold["fold"]), root / f"fold{outer_fold['fold']}"
        if fold not in selected:
            continue
        fold_dir.mkdir(parents=True, exist_ok=True)
        cls_path, seg_path = fold_dir / "oof_classification.csv", fold_dir / "oof_segmentation_cases.csv"
        if cfg.resume and cls_path.exists() and seg_path.exists():
            all_cls.extend(pd.read_csv(cls_path, dtype={"pid": str}).to_dict("records")); all_seg.extend(pd.read_csv(seg_path, dtype={"pid": str}).to_dict("records")); fold_rows.append(json.loads((fold_dir / "selection.json").read_text(encoding="utf-8"))); continue
        item = inner_by_fold[fold]
        train_records = records_for_pids(records, outer_fold["train_pids"])
        test_records = records_for_pids(records, outer_fold["val_pids"])
        dev_train_records, dev_records = records_for_pids(records, item["train_pids"]), records_for_pids(records, item["dev_pids"])
        selection_path = fold_dir / "selection.json"
        if cfg.resume and selection_path.exists():
            info = json.loads(selection_path.read_text(encoding="utf-8"))
            selected_epoch = int(info["selected_epoch"])
        else:
            selected_epoch, selection_score = _select_epoch(cfg, dev_train_records, dev_records, clinical, names, device, fold_dir, fold, fixed_categories)
            info = {
                "fold": fold,
                "selected_epoch": selected_epoch,
                "selection_score": selection_score,
                "clinical_preprocessing": provenance["folds"][str(fold)],
            }
            selection_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        fold_rows.append(info)
        cls_rows, seg_rows = _final_predict(cfg, train_records, test_records, clinical, names, selected_epoch, device, fold_dir, fold, fixed_categories)
        pd.DataFrame(cls_rows).to_csv(cls_path, index=False, encoding="utf-8-sig"); pd.DataFrame(seg_rows).to_csv(seg_path, index=False, encoding="utf-8-sig")
        all_cls.extend(cls_rows); all_seg.extend(seg_rows)
    cls_oof, seg_oof = normalize_oof_frame(all_cls), normalize_oof_frame(all_seg)
    expected_records = records_for_pids(
        records,
        [pid for item in outer["folds"] if int(item["fold"]) in selected for pid in item["val_pids"]],
    )
    validate_multitask_oof(cls_oof, seg_oof, expected_records)
    cls_dir, seg_dir = Path(cfg.output_root) / cfg.dataset / "classification" / cfg.method, Path(cfg.output_root) / cfg.dataset / "segmentation" / cfg.method
    cls_dir.mkdir(parents=True, exist_ok=True); seg_dir.mkdir(parents=True, exist_ok=True)
    cls_oof.to_csv(cls_dir / "oof_classification.csv", index=False, encoding="utf-8-sig"); seg_oof.to_csv(seg_dir / "oof_segmentation_cases.csv", index=False, encoding="utf-8-sig")
    probability_columns = [f"proba_{name}" for name in names]
    cls_metrics = classification_metrics(cls_oof["y_true"], cls_oof[probability_columns].to_numpy(), names); seg_metrics = {metric: float(seg_oof[metric].mean()) for metric in ("Dice", "mIoU", "HD95", "ASSD")}
    save_per_class_metrics(cls_dir / "per_class_metrics.csv", cls_metrics["per_class"])
    cls_flat = {key: value for key, value in cls_metrics.items() if key != "per_class"}
    for name, values in cls_metrics["per_class"].items(): cls_flat.update({f"class_{name}_{key}": value for key, value in values.items()})
    metadata = MODEL_METADATA[cfg.method]
    common = {"dataset": cfg.dataset, "method": cfg.method, "display_name": metadata["display_name"], "implementation": metadata["implementation"], "source": metadata["source"], "version": metadata["version"], "upstream_version": metadata["upstream_version"], "interactive": False, "eligible_for_automatic_comparison": metadata["eligible_for_automatic_comparison"], "protocol_version": PROTOCOL_VERSION, "fold_fingerprint": outer["split_fingerprint"], "inner_fold_fingerprint": inner["split_fingerprint"]}
    (cls_dir / "summary.json").write_text(json.dumps({**common, "task": "classification", **{key: round(float(value), 3) for key, value in cls_flat.items()}}, ensure_ascii=False, indent=2), encoding="utf-8")
    (seg_dir / "summary.json").write_text(json.dumps({**common, "task": "segmentation", **{key: round(float(value), 3) for key, value in seg_metrics.items()}}, ensure_ascii=False, indent=2), encoding="utf-8")
    cls_fold_metrics = pd.DataFrame([
        {"fold": int(fold), **{key: value for key, value in classification_metrics(group["y_true"], group[probability_columns].to_numpy(), names).items() if key != "per_class"}}
        for fold, group in cls_oof.groupby("fold")
    ])
    seg_fold_metrics = seg_oof.groupby("fold")[["Dice", "mIoU", "HD95", "ASSD"]].mean().reset_index()
    pd.DataFrame(fold_rows).to_csv(root / "selection_per_fold.csv", index=False, encoding="utf-8-sig")
    cls_fold_metrics.to_csv(cls_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    seg_fold_metrics.to_csv(seg_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary_rows = []
    for task_name, frame, metrics in (
        ("classification", cls_fold_metrics, ["ACC", "Macro-F1", "Macro-AUC", "Macro-Sensitivity", "Macro-Specificity"]),
        ("segmentation", seg_fold_metrics, ["Dice", "mIoU", "HD95", "ASSD"]),
    ):
        for metric in metrics:
            mean, std = float(frame[metric].mean()), float(frame[metric].std(ddof=1))
            summary_rows.append({"task": task_name, "metric": metric, "mean": mean, "std": std, "mean_std": f"{mean:.3f} ± {std:.3f}"})
    pd.DataFrame(summary_rows).to_csv(root / "fivefold_mean_std.csv", index=False, encoding="utf-8-sig")
    (root / "run_config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"classification": cls_metrics, "segmentation": seg_metrics}
