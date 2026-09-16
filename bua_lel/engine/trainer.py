import os
import sys
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]


from bua_lel.data import DualTaskDataset
from bua_lel.data.paired_transforms import build_training_transform
from bua_lel.models.bua_lel import BUALEL
from bua_lel.models.medsam_mtl import MedSAMMTLModel

from bua_lel.utils.seed import seed_everything
from bua_lel.data.preprocessing import (
    build_pid_and_labels,
    fit_fold_cat_maps,
    fit_fold_num_scaler,
    read_excel_df,
    subset_by_pid_set,
)
from bua_lel.utils.optim import (
    freeze_bn_running_stats,
    build_optimizer_param_groups,
)
from bua_lel.utils.roc import plot_multiclass_roc

from bua_lel.engine.losses import make_nomissing_loss_fn
from bua_lel.engine.train_eval import train_one_epoch, evaluate
from baselines.splits import (
    create_or_load_folds,
    create_or_load_group_folds,
    create_or_load_input_order_folds,
)
from baselines.metrics import classification_metrics
from baselines.results import save_per_class_metrics


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    image_dir: str = str(PROJECT_ROOT / "data" / "images")
    mask_dir: str = str(PROJECT_ROOT / "data" / "masks")
    clinical_excel: str = str(PROJECT_ROOT / "data" / "clinical.xlsx")

    save_dir: str = "outputs/bua_lel"
    num_workers: int = 4

    folds: int = 5


    selected_folds: Tuple[int, ...] = ()
    epochs: int = 50
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    early_stop_patience: int = 12


    selection_policy: str = "fixed_final_epoch"

    use_pca: bool = False
    pca_dim: int = 100
    clinical_embed_dim: int = 128
    detach_roi_in_cls: bool = True
    detach_segfeat_in_cls: bool = False


    lambda_cons: float = 0.0


    use_boundary_refiner: bool = True
    boundary_num_nodes: int = 64
    boundary_hidden_dim: int = 128
    boundary_gnn_layers: int = 3
    boundary_max_offset: float = 0.08
    lambda_anchor: float = 0.0
    boundary_token_gate_init: float = -2.0
    use_uncertainty_weighting: bool = False


    boundary_radius_ratio: float = 0.15
    peritumor_radius_ratio: float = 0.40
    min_boundary_radius: int = 2
    max_boundary_radius: int = 4
    min_peritumor_radius: int = 4
    max_peritumor_radius: int = 8

    force_full_observed_mask: bool = True


    unknown_cat_as_missing: bool = False
    clinical_exclude_columns: Tuple[str, ...] = ()
    clinical_preprocessing_profile: str = "missing_aware"
    image_mode: str = "grayscale"
    fold_group_column: Optional[str] = None


    medsam_checkpoint_path: Optional[str] = str(PROJECT_ROOT / "checkpoints" / "medsam_vit_b.pth")
    medsam_model_type: str = "vit_b"
    freeze_medsam: bool = True
    unfreeze_medsam_last_n: int = 0
    medsam_activation_checkpointing: bool = False

    image_size: int = 256


    medsam_input_size: Optional[int] = 1024
    architecture_profile: str = "extended"
    model_family: str = "bua_lel"
    training_task: str = "joint"
    visual_backbone: str = "medsam_vit_b"
    visual_checkpoint_path: Optional[str] = None
    backbone_unfreeze_policy: str = "last_transformer_block"


    scheduler: str = "cosine"
    min_lr: float = 1e-6


    use_class_weight: bool = True
    num_classes: int = 3
    label_smoothing: float = 0.05


    seg_thr: float = 0.5
    seg_bce_weight: float = 0.5

    seg_dice_weight: Optional[float] = None
    seg_boundary_weight: float = 0.0
    seg_coarse_weight: float = 0.0
    train_augmentation_profile: str = "none"


    score_w_seg: float = 0.3
    score_w_cls: float = 0.7


    freeze_bn: bool = True
    grad_accum_steps: int = 1


    cls_lr_mult: float = 2.0
    cls_name_keywords: Tuple[str, ...] = (
        "cls_head", "classifier", "logit", "fc",
        "gating", "clin_module", "phi_clin", "img_gate", "clin_gate",


        "cls_fusion", "clsfeat_to_imgdim", "feature_fusion",
        "hetero_graph", "alternative_fusion", "ultrasound_only_fusion",
        "clinical_only_fusion", "clin_graph"
        , "image_projection", "clinical_projection"
    )


    save_fold_metrics_csv: bool = True
    summary_table_name: str = "paper_table_mean_std.csv"
    save_final_oof_roc: bool = True
    final_roc_name: str = "OOF_ROC.png"
    save_oof_npz: bool = False
    keep_fold_checkpoints: bool = True
    fold_manifest_path: str = str(PROJECT_ROOT / "outputs" / "splits" / "HER2USC" / "classification_seed42_5fold.json")
    fold_manifest_strategy: str = "canonical_sorted"
    experiment_name: str = "full"
    dataset_name: str = "HER2USC"
    benchmark_output_root: str = str(PROJECT_ROOT / "outputs")


    modality: str = "multimodal"
    enabled_regions: Tuple[str, ...] = ("core", "boundary", "peritumoral")
    use_z_geo: bool = True
    use_z_unc: bool = True
    fusion_mode: str = "heterog"
    late_raw_clinical_fusion: bool = False
    use_boundary_ambiguity: bool = True
    use_anchor_graph: bool = True
    use_zonal_evidence: bool = True


def composite_score(stats: Dict[str, float], w_seg=0.5, w_cls=0.5) -> float:
    return w_seg * float(stats.get("dice", 0.0)) + w_cls * float(stats.get("macro_f1", 0.0))


VALID_SELECTION_POLICIES = {"fixed_final_epoch", "best_composite_early_stop"}
VALID_FOLD_MANIFEST_STRATEGIES = {
    "canonical_sorted", "input_order", "stratified_group"
}


def validate_train_config(cfg: TrainConfig) -> None:
    if cfg.selection_policy not in VALID_SELECTION_POLICIES:
        raise ValueError(
            f"Unknown selection_policy={cfg.selection_policy!r}; "
            f"expected one of {sorted(VALID_SELECTION_POLICIES)}"
        )
    if cfg.fold_manifest_strategy not in VALID_FOLD_MANIFEST_STRATEGIES:
        raise ValueError(
            f"Unknown fold_manifest_strategy={cfg.fold_manifest_strategy!r}; "
            f"expected one of {sorted(VALID_FOLD_MANIFEST_STRATEGIES)}"
        )
    if cfg.selection_policy == "best_composite_early_stop" and cfg.early_stop_patience < 1:
        raise ValueError("best_composite_early_stop requires early_stop_patience >= 1")
    if cfg.train_augmentation_profile not in {
        "none", "ultrasound_mild", "ultrasound_gentle", "dermoscopy_mild"
    }:
        raise ValueError(
            "train_augmentation_profile must be one of 'none', 'ultrasound_mild', "
            "'ultrasound_gentle', or 'dermoscopy_mild'"
        )
    dice_weight = (
        1.0 - float(cfg.seg_bce_weight)
        if cfg.seg_dice_weight is None
        else float(cfg.seg_dice_weight)
    )
    seg_weights = (
        float(cfg.seg_bce_weight), dice_weight,
        float(cfg.seg_boundary_weight), float(cfg.seg_coarse_weight),
    )
    if any(weight < 0.0 for weight in seg_weights) or not np.isclose(sum(seg_weights), 1.0):
        raise ValueError("segmentation loss weights must be non-negative and sum to 1")
    if cfg.selected_folds:
        selected = tuple(int(fold) for fold in cfg.selected_folds)
        if len(set(selected)) != len(selected) or any(fold < 1 or fold > cfg.folds for fold in selected):
            raise ValueError(
                f"selected_folds must contain unique fold identifiers in 1..{cfg.folds}, "
                f"got {cfg.selected_folds!r}"
            )
    if cfg.model_family not in {"bua_lel", "medsam_standard_decoder_fusion"}:
        raise ValueError(f"unsupported model_family: {cfg.model_family!r}")
    if cfg.training_task not in {"joint", "classification_only", "segmentation_only"}:
        raise ValueError(f"unsupported training_task: {cfg.training_task!r}")
    if cfg.training_task == "classification_only" and cfg.modality != "clinical_only":
        raise ValueError("classification_only is reserved for the strict clinical_only ablation")
    if cfg.visual_backbone not in {
        "medsam_vit_b", "sam_vit_b", "imagenet_resnet50",
    }:
        raise ValueError(f"unsupported visual_backbone: {cfg.visual_backbone!r}")
    expected_policy = {
        "medsam_vit_b": "last_transformer_block",
        "sam_vit_b": "last_transformer_block",
        "imagenet_resnet50": "last_bottleneck",
    }[cfg.visual_backbone]
    if cfg.backbone_unfreeze_policy != expected_policy:
        raise ValueError(
            f"{cfg.visual_backbone} requires backbone_unfreeze_policy={expected_policy!r}"
        )


def create_fold_manifest(
    cfg: TrainConfig,
    pids: List[str],
    labels: np.ndarray,
) -> Dict[str, object]:
    """Resolve the configured split policy without allowing a silent reorder."""
    path = Path(cfg.fold_manifest_path)
    if cfg.fold_manifest_strategy == "canonical_sorted":
        return create_or_load_folds(path, pids, labels, n_splits=cfg.folds, seed=cfg.seed)
    if cfg.fold_manifest_strategy == "input_order":
        return create_or_load_input_order_folds(
            path, pids, labels, n_splits=cfg.folds, seed=cfg.seed
        )
    if cfg.fold_manifest_strategy == "stratified_group":
        if not cfg.fold_group_column:
            raise ValueError("stratified_group requires fold_group_column")
        table = read_excel_df(cfg.clinical_excel)
        pid_column = table.columns[0]
        if cfg.fold_group_column not in table.columns:
            raise ValueError(f"missing fold group column: {cfg.fold_group_column}")
        group_by_pid = dict(zip(table[pid_column].astype(str), table[cfg.fold_group_column].astype(str)))
        groups = [group_by_pid[str(pid)] for pid in pids]
        return create_or_load_group_folds(
            path, pids, labels, groups, n_splits=cfg.folds, seed=cfg.seed
        )
    raise AssertionError(f"validated strategy unexpectedly unsupported: {cfg.fold_manifest_strategy}")


def validate_oof_case_coverage(
    oof_case_rows: List[Dict[str, object]],
    all_pids: List[str],
) -> pd.DataFrame:
    """Fail closed unless final OOF records cover each fixed-fold patient once."""
    frame = pd.DataFrame(oof_case_rows)
    if frame.empty:
        raise RuntimeError("Final OOF evaluation did not return any patient records")
    if "pid" not in frame.columns:
        raise RuntimeError("Final OOF evaluation records are missing pid")
    duplicated = frame["pid"].duplicated(keep=False)
    if duplicated.any():
        raise RuntimeError(f"Duplicate OOF pid(s): {frame.loc[duplicated, 'pid'].tolist()[:10]}")
    if set(map(str, frame["pid"])) != set(map(str, all_pids)):
        raise RuntimeError("OOF patients do not exactly match the fixed-fold cohort")
    return frame


def advance_checkpoint_selection(
    best_score: float,
    best_epoch: int,
    stale_epochs: int,
    *,
    score: float,
    epoch: int,
    patience_limit: int,
) -> Tuple[Tuple[float, int, int], bool, bool]:
    """Update strict-improvement early-stop state for one validation epoch."""
    if not np.isfinite(score):
        raise ValueError(f"checkpoint selection received a non-finite score at epoch {epoch}: {score}")
    if patience_limit < 1:
        raise ValueError("patience_limit must be >= 1")
    if score > best_score:
        return (float(score), int(epoch), 0), True, False
    stale_epochs = int(stale_epochs) + 1
    return (float(best_score), int(best_epoch), stale_epochs), False, stale_epochs >= patience_limit


def build_model(cfg: TrainConfig, fold_ds: DualTaskDataset) -> nn.Module:
    """Build a model from TrainConfig; shared by training and checkpoint analysis."""
    if cfg.model_family == "medsam_standard_decoder_fusion":
        if cfg.late_raw_clinical_fusion:
            raise ValueError("standard MedSAM fusion must not append raw clinical features twice")
        if cfg.lambda_anchor != 0.0:
            raise ValueError("standard MedSAM fusion requires lambda_anchor=0")
        return MedSAMMTLModel(
            clinical_dim=fold_ds.get_feature_dim(),
            num_classes=cfg.num_classes,
            medsam_checkpoint_path=cfg.medsam_checkpoint_path,
            medsam_model_type=cfg.medsam_model_type,
            freeze_medsam=cfg.freeze_medsam,
            unfreeze_medsam_last_n=cfg.unfreeze_medsam_last_n,
            medsam_activation_checkpointing=cfg.medsam_activation_checkpointing,
            image_size=cfg.medsam_input_size or 1024,
            cls_dropout=0.3,
        )
    return BUALEL(
        clinical_dim=fold_ds.get_feature_dim(),
        numeric_slice=fold_ds.numeric_slice,
        onehot_slices_dict=fold_ds.onehot_slices,
        num_classes=cfg.num_classes,
        use_pca=cfg.use_pca,
        pca_dim=cfg.pca_dim,
        clin_embed_dim=cfg.clinical_embed_dim,
        detach_roi_in_cls=cfg.detach_roi_in_cls,
        detach_segfeat_in_cls=cfg.detach_segfeat_in_cls,
        lambda_cons=cfg.lambda_cons,
        use_boundary_refiner=cfg.use_boundary_refiner,
        boundary_num_nodes=cfg.boundary_num_nodes,
        boundary_hidden_dim=cfg.boundary_hidden_dim,
        boundary_gnn_layers=cfg.boundary_gnn_layers,
        boundary_max_offset=cfg.boundary_max_offset,
        lambda_anchor=cfg.lambda_anchor,
        boundary_token_gate_init=cfg.boundary_token_gate_init,
        boundary_radius_ratio=cfg.boundary_radius_ratio,
        peritumor_radius_ratio=cfg.peritumor_radius_ratio,
        min_boundary_radius=cfg.min_boundary_radius,
        max_boundary_radius=cfg.max_boundary_radius,
        min_peritumor_radius=cfg.min_peritumor_radius,
        max_peritumor_radius=cfg.max_peritumor_radius,
        medsam_checkpoint_path=cfg.medsam_checkpoint_path,
        medsam_model_type=cfg.medsam_model_type,
        freeze_medsam=cfg.freeze_medsam,
        unfreeze_medsam_last_n=cfg.unfreeze_medsam_last_n,
        medsam_activation_checkpointing=cfg.medsam_activation_checkpointing,
        image_size=cfg.medsam_input_size or cfg.image_size,
        modality=cfg.modality,
        enabled_regions=cfg.enabled_regions,
        use_z_geo=cfg.use_z_geo,
        use_z_unc=cfg.use_z_unc,
        fusion_mode=cfg.fusion_mode,
        late_raw_clinical_fusion=cfg.late_raw_clinical_fusion,
        use_boundary_ambiguity=cfg.use_boundary_ambiguity,
        use_anchor_graph=cfg.use_anchor_graph,
        use_zonal_evidence=cfg.use_zonal_evidence,
        architecture_profile=cfg.architecture_profile,
        visual_backbone=cfg.visual_backbone,
        visual_checkpoint_path=cfg.visual_checkpoint_path,
        backbone_unfreeze_policy=cfg.backbone_unfreeze_policy,
    )


def main(cfg: TrainConfig):
    validate_train_config(cfg)
    os.makedirs(cfg.save_dir, exist_ok=True)
    pd.DataFrame([cfg.__dict__]).to_csv(
        os.path.join(cfg.save_dir, "experiment_config.csv"),
        index=False, encoding="utf-8-sig", float_format="%.9g",
    )
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)

    all_pids, y_all = build_pid_and_labels(cfg.image_dir, cfg.clinical_excel)
    print(f"[Split] total matched (image ∩ clinical) = {len(all_pids)}")

    manifest = create_fold_manifest(cfg, all_pids, y_all)
    pid_to_label = {str(pid): int(label) for pid, label in zip(all_pids, y_all)}
    assignment_rows = [
        {"pid": str(pid), "label": pid_to_label[str(pid)], "fold": int(split["fold"])}
        for split in manifest["folds"] for pid in split["val_pids"]
    ]
    pd.DataFrame(assignment_rows).sort_values("pid").to_csv(
        os.path.join(cfg.save_dir, "fold_assignments.csv"),
        index=False, encoding="utf-8-sig",
    )

    selected_fold_set = set(map(int, cfg.selected_folds)) or {
        int(split["fold"]) for split in manifest["folds"]
    }
    expected_oof_pids = {
        str(pid)
        for split in manifest["folds"]
        if int(split["fold"]) in selected_fold_set
        for pid in split["val_pids"]
    }
    print(f"[Split] running folds={sorted(selected_fold_set)}; expected OOF patients={len(expected_oof_pids)}")

    best_rows = []
    oof_true_all: List[np.ndarray] = []
    oof_proba_all: List[np.ndarray] = []
    oof_case_rows: List[Dict[str, object]] = []
    unified_fold_rows: List[Dict[str, object]] = []

    for split in manifest["folds"]:
        fold = int(split["fold"])
        if fold not in selected_fold_set:
            continue
        print(f"\n================= Fold {fold}/{cfg.folds} =================")

        train_pids = set(map(str, split["train_pids"]))
        val_pids = set(map(str, split["val_pids"]))

        cat_maps = fit_fold_cat_maps(
            cfg.clinical_excel,
            train_pids,
            cfg.clinical_exclude_columns,
            clinical_preprocessing_profile=cfg.clinical_preprocessing_profile,
        )
        num_scaler = fit_fold_num_scaler(cfg.clinical_excel, train_pids, cfg.clinical_exclude_columns)

        dataset_kwargs = dict(
            image_dir=cfg.image_dir,
            mask_dir=cfg.mask_dir,
            clinical_excel=cfg.clinical_excel,
            mode="seg",
            allow_missing_mask=True,
            return_pid=True,
            num_scaler=num_scaler,
            cat_maps=cat_maps,
            unknown_cat_as_missing=cfg.unknown_cat_as_missing,
            return_cat_targets=True,
            exclude_feature_columns=list(cfg.clinical_exclude_columns),
            clinical_preprocessing_profile=cfg.clinical_preprocessing_profile,
            image_mode=cfg.image_mode,
            image_size=(cfg.image_size, cfg.image_size),
        )

        fold_ds = DualTaskDataset(transform=None, **dataset_kwargs)
        training_transform = build_training_transform(cfg.train_augmentation_profile)
        train_fold_ds = (
            fold_ds
            if training_transform is None
            else DualTaskDataset(transform=training_transform, **dataset_kwargs)
        )

        train_set = subset_by_pid_set(train_fold_ds, train_pids)
        val_set = subset_by_pid_set(fold_ds, val_pids)

        train_loader = DataLoader(
            train_set,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model = build_model(cfg, fold_ds).to(device)

        image_encoder = model.encoder.image_encoder
        backbone_total_params = sum(p.numel() for p in image_encoder.parameters())
        backbone_trainable_params = sum(
            p.numel() for p in image_encoder.parameters() if p.requires_grad
        )
        freeze_all_medsam = (
            cfg.visual_backbone in {"medsam_vit_b", "sam_vit_b"}
            and cfg.freeze_medsam and cfg.unfreeze_medsam_last_n == 0
        )
        if freeze_all_medsam and backbone_trainable_params != 0:
            raise RuntimeError(
                "Freeze-all invariant failed: image encoder still has "
                f"{backbone_trainable_params:,} trainable parameters"
            )
        trainable_backbone_names = {
            name for name, parameter in image_encoder.named_parameters()
            if parameter.requires_grad
        }
        if cfg.visual_backbone == "imagenet_resnet50":
            if not trainable_backbone_names or not all(
                name.startswith("layer4.2.") for name in trainable_backbone_names
            ):
                raise RuntimeError(
                    "ResNet-50 invariant failed: only layer4[-1] may be trainable"
                )


        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Model] total params = {total_params:,}")
        print(f"[Model] trainable params = {trainable_params:,}")
        print(
            f"[Visual backbone] name={cfg.visual_backbone}, "
            f"checkpoint={cfg.visual_checkpoint_path or cfg.medsam_checkpoint_path}, "
            f"freeze={cfg.freeze_medsam}, "
            f"unfreeze_last_n={cfg.unfreeze_medsam_last_n}, "
            f"policy={cfg.backbone_unfreeze_policy}, "
            f"dataset_image_size={cfg.image_size}, "
            f"encoder_input_size={cfg.medsam_input_size or cfg.image_size}"
        )
        print(
            f"[Visual backbone audit] image_encoder_total={backbone_total_params:,}, "
            f"image_encoder_trainable={backbone_trainable_params:,}, "
            f"freeze_all_confirmed={freeze_all_medsam and backbone_trainable_params == 0}"
        )
        print(
            f"[Boundary] use={cfg.use_boundary_refiner}, "
            f"nodes={cfg.boundary_num_nodes}, hidden={cfg.boundary_hidden_dim}, "
            f"layers={cfg.boundary_gnn_layers}, max_offset={cfg.boundary_max_offset}, "
            f"lambda_anchor={cfg.lambda_anchor}, "
            f"gate_init={cfg.boundary_token_gate_init}, "
            f"uncertainty_weighting={cfg.use_uncertainty_weighting}"
        )
        print(
            f"[ROI] boundary_ratio={cfg.boundary_radius_ratio}, "
            f"peritumor_ratio={cfg.peritumor_radius_ratio}, "
            f"boundary_clip=[{cfg.min_boundary_radius},{cfg.max_boundary_radius}], "
            f"peritumor_clip=[{cfg.min_peritumor_radius},{cfg.max_peritumor_radius}]"
        )


        if cfg.use_class_weight:
            train_labels = np.asarray([pid_to_label[pid] for pid in train_pids], dtype=np.int64)
            counts = np.bincount(train_labels, minlength=cfg.num_classes).astype(np.float32)
            weights = (counts.sum() / (counts + 1e-6))
            weights = weights / weights.mean()
            class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
            cls_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
            print("class_weights:", weights)
        else:
            cls_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)


        groups, (n_other, n_cls) = build_optimizer_param_groups(
            model=model,
            base_lr=cfg.lr,
            cls_lr_mult=cfg.cls_lr_mult,
            weight_decay=cfg.weight_decay,
            cls_name_keywords=tuple(k.lower() for k in cfg.cls_name_keywords),
        )
        optimizer = torch.optim.AdamW(groups)
        frozen_backbone_param_ids = {
            id(parameter) for parameter in image_encoder.parameters()
            if not parameter.requires_grad
        }
        optimizer_param_ids = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        if frozen_backbone_param_ids & optimizer_param_ids:
            raise RuntimeError("Frozen visual-backbone parameters entered the optimizer")
        print(f"optimizer groups: other={n_other} params, cls/fusion={n_cls} params, cls_lr_mult={cfg.cls_lr_mult}")


        if cfg.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)
        elif cfg.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        else:
            scheduler = None

        scaler_amp = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        fold_dir = os.path.join(cfg.save_dir, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        epoch_metrics: List[Dict[str, float]] = []
        best_score = -1e9
        best_epoch = -1
        best_path = os.path.join(cfg.save_dir, f"best_fold{fold}.pth")
        patience = 0

        loss_fn = make_nomissing_loss_fn(
            cls_criterion=cls_criterion,
            seg_bce_weight=cfg.seg_bce_weight,
            seg_dice_weight=cfg.seg_dice_weight,
            seg_boundary_weight=cfg.seg_boundary_weight,
            seg_coarse_weight=cfg.seg_coarse_weight,
            force_full_observed_mask=cfg.force_full_observed_mask,
            use_uncertainty_weighting=cfg.use_uncertainty_weighting,
            lambda_anchor=cfg.lambda_anchor,
            classification_only=cfg.training_task == "classification_only",
            segmentation_only=cfg.training_task == "segmentation_only",
        )

        def save_selected_checkpoint(epoch: int, selection_score: Optional[float] = None) -> None:
            payload = {
                "fold": fold,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "selection_policy": cfg.selection_policy,
                "cfg": cfg.__dict__,
                "cat_maps": cat_maps,
                "num_scaler": num_scaler,
                "clinical_dim": fold_ds.get_feature_dim(),
                "feature_names": getattr(fold_ds, "feature_names", None),
                "numeric_slice": getattr(fold_ds, "numeric_slice", None),
                "onehot_slices": getattr(fold_ds, "onehot_slices", None),
            }
            if selection_score is not None:
                payload["best_score"] = float(selection_score)
            torch.save(payload, best_path)

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()

            if cfg.freeze_bn:
                freeze_bn_running_stats(model)

            lr_group0 = optimizer.param_groups[0]["lr"]
            lr_group1 = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else lr_group0

            train_stats = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                scaler_amp=scaler_amp,
                loss_fn=loss_fn,
                max_grad_norm=cfg.max_grad_norm,
                grad_accum_steps=max(1, cfg.grad_accum_steps),
            )

            val_stats = None
            selection_score = None
            if cfg.selection_policy == "best_composite_early_stop":


                val_stats = evaluate(
                    model=model,
                    loader=val_loader,
                    device=device,
                    num_classes=cfg.num_classes,
                    seg_thr=cfg.seg_thr,
                    cls_criterion=cls_criterion,
                    seg_bce_weight=cfg.seg_bce_weight,
                    force_full_observed_mask=cfg.force_full_observed_mask,
                    return_preds=False,
                    compute_surface_metrics=False,
                    return_case_records=False,
                    classification_only=cfg.training_task == "classification_only",
            segmentation_only=cfg.training_task == "segmentation_only",
                    seg_dice_weight=cfg.seg_dice_weight,
                    seg_boundary_weight=cfg.seg_boundary_weight,
                    seg_coarse_weight=cfg.seg_coarse_weight,
                )
                selection_score = (
                    float(val_stats["macro_f1"])
                    if cfg.training_task == "classification_only"
                    else composite_score(val_stats, w_seg=cfg.score_w_seg, w_cls=cfg.score_w_cls)
                )

            if scheduler is not None:
                scheduler.step()

            dt = time.time() - t0
            if val_stats is None:
                print(
                    f"Epoch {epoch:03d} | lr(backbone) {lr_group0:.2e} lr(cls/fusion) {lr_group1:.2e} | "
                    f"train: total {train_stats['total_loss']:.4f} seg {train_stats['seg_loss']:.4f} "
                    f"cls {train_stats['cls_loss']:.4f} | time {dt:.1f}s"
                )
            else:
                print(
                    f"Epoch {epoch:03d} | lr(backbone) {lr_group0:.2e} lr(cls/fusion) {lr_group1:.2e} | "
                    f"train: total {train_stats['total_loss']:.4f} seg {train_stats['seg_loss']:.4f} "
                    f"cls {train_stats['cls_loss']:.4f} | "
                    f"val: dice {val_stats['dice']:.4f} miou {val_stats['miou']:.4f} "
                    f"acc {val_stats['acc']:.4f} f1 {val_stats['macro_f1']:.4f} "
                    f"auc {val_stats['auc_macro_ovr']:.4f} | score {selection_score:.4f} | time {dt:.1f}s"
                )

            row = {"fold": fold, "epoch": epoch, "lr_backbone": lr_group0, "lr_cls": lr_group1}
            row.update(train_stats)
            if val_stats is not None:
                row.update(val_stats)
                row["score"] = selection_score
            epoch_metrics.append(row)

            if cfg.selection_policy == "best_composite_early_stop":
                (best_score, best_epoch, patience), improved, should_stop = advance_checkpoint_selection(
                    best_score,
                    best_epoch,
                    patience,
                    score=float(selection_score),
                    epoch=epoch,
                    patience_limit=cfg.early_stop_patience,
                )
                if improved:
                    save_selected_checkpoint(epoch, selection_score=best_score)
                    print(f"✅ Saved best checkpoint to {best_path} (score={best_score:.4f})")
                elif should_stop:
                    print(f"🛑 Early stopping: patience={cfg.early_stop_patience}")
                    break
            elif epoch == cfg.epochs:
                best_epoch = epoch
                save_selected_checkpoint(epoch)
                print(f"✅ Saved fixed final epoch to {best_path}")

        if best_epoch < 1 or not os.path.exists(best_path):
            raise RuntimeError(f"Fold {fold} did not produce a selected checkpoint")

        if cfg.save_fold_metrics_csv:
            df_fold = pd.DataFrame(epoch_metrics)
            csv_path = os.path.join(fold_dir, f"fold{fold}_metrics.csv")
            df_fold.to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.9g")
            print(f"📄 Saved fold metrics csv -> {csv_path}")


        best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        checkpoint_policy = best_checkpoint.get("selection_policy", "best_composite_early_stop")
        if checkpoint_policy != cfg.selection_policy:
            raise RuntimeError(
                f"Fold {fold} checkpoint selection policy mismatch: "
                f"{checkpoint_policy!r} != {cfg.selection_policy!r}"
            )
        model.load_state_dict(best_checkpoint["model_state"], strict=True)
        best_eval = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            num_classes=cfg.num_classes,
            seg_thr=cfg.seg_thr,
            cls_criterion=cls_criterion,
            seg_bce_weight=cfg.seg_bce_weight,
            force_full_observed_mask=cfg.force_full_observed_mask,
            return_preds=True,
            compute_surface_metrics=True,
            return_case_records=True,
            classification_only=cfg.training_task == "classification_only",
            segmentation_only=cfg.training_task == "segmentation_only",
            seg_dice_weight=cfg.seg_dice_weight,
            seg_boundary_weight=cfg.seg_boundary_weight,
            seg_coarse_weight=cfg.seg_coarse_weight,
        )
        final_checkpoint_score = (
            float(best_eval["macro_f1"])
            if cfg.training_task == "classification_only"
            else composite_score(best_eval, w_seg=cfg.score_w_seg, w_cls=cfg.score_w_cls)
        )
        saved_selection_score = best_checkpoint.get("best_score")
        if cfg.selection_policy == "best_composite_early_stop":
            if saved_selection_score is None or not np.isfinite(float(saved_selection_score)):
                raise RuntimeError(f"Fold {fold} early-stop checkpoint is missing a finite best_score")
            if not np.isclose(float(saved_selection_score), final_checkpoint_score, rtol=0.0, atol=1e-6):
                raise RuntimeError(
                    f"Fold {fold} best-score mismatch: saved={saved_selection_score}, "
                    f"recomputed={final_checkpoint_score}"
                )
        best_score = final_checkpoint_score
        oof_true_all.append(best_eval["_y_true"])
        oof_proba_all.append(best_eval["_y_proba"])
        fold_case_rows = best_eval["_case_records"]
        for row in fold_case_rows:
            row["fold"] = fold
            row["experiment"] = cfg.experiment_name
        oof_case_rows.extend(fold_case_rows)
        pd.DataFrame(fold_case_rows).to_csv(
            os.path.join(fold_dir, "oof_cases.csv"), index=False,
            encoding="utf-8-sig", float_format="%.9g",
        )
        fold_frame = pd.DataFrame(fold_case_rows)
        fold_probs = fold_frame[[f"prob_c{c}" for c in range(cfg.num_classes)]].to_numpy()
        fold_cls = classification_metrics(fold_frame["y_true"], fold_probs)
        fold_unified = {
            "experiment": cfg.experiment_name,
            "fold": fold,
            **{k: v for k, v in fold_cls.items() if k != "per_class"},
            **fold_frame[["Dice", "mIoU", "HD95", "ASSD"]].mean().to_dict(),
            "segmentation_applicable": cfg.training_task != "classification_only",
        }
        unified_fold_rows.append(fold_unified)

        df_fold = pd.DataFrame(epoch_metrics)
        best_epoch_rows = df_fold.loc[df_fold["epoch"] == best_epoch]
        if best_epoch_rows.empty:
            raise RuntimeError(f"Fold {fold} is missing metrics for selected epoch {best_epoch}")
        best_row = best_epoch_rows.iloc[-1].to_dict()
        best_row.update({k: v for k, v in best_eval.items() if not k.startswith("_")})
        best_row["score"] = final_checkpoint_score
        best_row["selection_score"] = (
            float(saved_selection_score)
            if saved_selection_score is not None else final_checkpoint_score
        )
        best_row["selection_policy"] = cfg.selection_policy
        best_row["best_epoch"] = best_epoch
        best_row["final_oof_evaluation_count"] = 1
        if device.type == "cuda":
            best_row["peak_gpu_memory_allocated_mib"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            best_row["peak_gpu_memory_reserved_mib"] = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        best_rows.append(best_row)
        print(f"Fold {fold} done. Best score={best_score:.4f} at epoch {best_epoch}")
        if not cfg.keep_fold_checkpoints:
            os.remove(best_path)
            print(f"Removed evaluated fold checkpoint to conserve disk: {best_path}")


    oof_frame = (
        validate_oof_case_coverage(oof_case_rows, sorted(expected_oof_pids))
        if oof_case_rows else None
    )


    if cfg.save_final_oof_roc and len(oof_true_all) > 0:
        y_true_oof = np.concatenate(oof_true_all, axis=0)
        y_proba_oof = np.concatenate(oof_proba_all, axis=0)

        final_roc_path = os.path.join(cfg.save_dir, cfg.final_roc_name)
        if len(selected_fold_set) == cfg.folds:
            roc_title = "ROC (5-fold Cross-Validation)"
        else:
            folds_label = ", ".join(str(fold) for fold in sorted(selected_fold_set))
            roc_title = f"ROC (exploratory fold {folds_label} selected_fold)"
        plot_multiclass_roc(
            y_true=y_true_oof,
            y_proba=y_proba_oof,
            num_classes=cfg.num_classes,
            save_path=final_roc_path,
            title=roc_title,
        )
        print(f"📈 Saved ONE final OOF ROC -> {final_roc_path}")

        if cfg.save_oof_npz:
            npz_path = os.path.join(cfg.save_dir, "oof_predictions.npz")
            np.savez(npz_path, y_true=y_true_oof, y_proba=y_proba_oof)
            print(f"💾 Saved OOF predictions -> {npz_path}")

    if oof_frame is not None:
        oof_frame.sort_values("pid").to_csv(
            os.path.join(cfg.save_dir, "oof_cases.csv"), index=False,
            encoding="utf-8-sig", float_format="%.9g",
        )

        prob_cols = [f"prob_c{c}" for c in range(cfg.num_classes)]
        cls_result = classification_metrics(oof_frame["y_true"], oof_frame[prob_cols].to_numpy())
        seg_result = oof_frame[["Dice", "mIoU", "HD95", "ASSD"]].mean().to_dict()
        unified = {k: v for k, v in cls_result.items() if k != "per_class"}
        unified.update(seg_result)
        unified["segmentation_applicable"] = cfg.training_task != "classification_only"
        pd.DataFrame([{"experiment": cfg.experiment_name, **unified}]).to_csv(
            os.path.join(cfg.save_dir, "oof_metrics.csv"), index=False,
            encoding="utf-8-sig", float_format="%.9g",
        )
        fold_metrics_frame = pd.DataFrame(unified_fold_rows)
        fold_metrics_frame.to_csv(
            os.path.join(cfg.save_dir, "fold_metrics.csv"), index=False,
            encoding="utf-8-sig", float_format="%.9g",
        )
        metric_names = [
            "ACC", "Macro-F1", "Macro-AUC", "Macro-Sensitivity", "Macro-Specificity",
            "Dice", "mIoU", "HD95", "ASSD",
        ]
        summary_row = {"experiment": cfg.experiment_name}
        summary_row["segmentation_applicable"] = cfg.training_task != "classification_only"
        for metric in metric_names:
            summary_row[f"{metric}_mean"] = fold_metrics_frame[metric].mean()


            summary_row[f"{metric}_std"] = (
                0.0 if len(selected_fold_set) == 1
                else fold_metrics_frame[metric].std(ddof=1)
            )
        pd.DataFrame([summary_row]).to_csv(
            os.path.join(cfg.save_dir, "summary_metrics.csv"), index=False,
            encoding="utf-8-sig", float_format="%.9g",
        )


        if cfg.experiment_name == "full":
            fingerprint = manifest["split_fingerprint"]
            cls_dir = Path(cfg.benchmark_output_root) / cfg.dataset_name / "classification" / "bua_lel"
            seg_dir = Path(cfg.benchmark_output_root) / cfg.dataset_name / "segmentation" / "bua_lel"
            cls_dir.mkdir(parents=True, exist_ok=True)
            seg_dir.mkdir(parents=True, exist_ok=True)
            cls_oof = oof_frame[["pid", "fold", "y_true", "y_pred"]].copy()
            for c in range(cfg.num_classes):
                cls_oof[f"proba_{c}"] = oof_frame[f"prob_c{c}"]
            cls_oof.to_csv(
                cls_dir / "oof_classification.csv", index=False,
                encoding="utf-8-sig",
            )
            np.savez(
                cls_dir / "oof_classification.npz",
                pid=cls_oof["pid"].astype(str).to_numpy(), fold=cls_oof["fold"].to_numpy(),
                y_true=cls_oof["y_true"].to_numpy(),
                y_proba=cls_oof[[f"proba_{c}" for c in range(cfg.num_classes)]].to_numpy(),
            )
            oof_frame[["pid", "fold", "Dice", "mIoU", "HD95", "ASSD"]].to_csv(
                seg_dir / "oof_segmentation_cases.csv", index=False,
                encoding="utf-8-sig",
            )
            cls_flat = {k: v for k, v in cls_result.items() if k != "per_class"}
            save_per_class_metrics(cls_dir / "per_class_metrics.csv", cls_result["per_class"])
            for name, values in cls_result["per_class"].items():
                cls_flat.update({f"class_{name}_{key}": value for key, value in values.items()})
            cls_summary = {"dataset": cfg.dataset_name, "task": "classification", "method": "bua_lel", "fold_fingerprint": fingerprint, **{k: round(float(v), 3) for k, v in cls_flat.items()}}
            seg_summary = {"dataset": cfg.dataset_name, "task": "segmentation", "method": "bua_lel", "fold_fingerprint": fingerprint, **{k: round(float(v), 3) for k, v in seg_result.items()}}
            (cls_dir / "summary.json").write_text(json.dumps(cls_summary, indent=2, ensure_ascii=False), encoding="utf-8")
            (seg_dir / "summary.json").write_text(json.dumps(seg_summary, indent=2, ensure_ascii=False), encoding="utf-8")


    df_best = pd.DataFrame(best_rows)

    paper_cols = [
        "dice", "miou", "hd95", "assd", "seg_precision", "seg_recall",
        "acc", "macro_f1", "auc_macro_ovr",
        "sens_macro", "spec_macro", "score"
    ]
    for c in range(cfg.num_classes):
        paper_cols += [f"sens_c{c}", f"spec_c{c}"]
    paper_cols = [c for c in paper_cols if c in df_best.columns]

    mean_vals = df_best[paper_cols].mean(numeric_only=True)
    std_vals = (
        pd.Series(0.0, index=paper_cols)
        if len(selected_fold_set) == 1
        else df_best[paper_cols].std(numeric_only=True, ddof=1)
    )

    summary = []
    for k in paper_cols:
        mval = mean_vals[k]
        sval = std_vals[k]
        if np.isnan(mval):
            summary.append({"metric": k, "mean": np.nan, "std": np.nan, "mean±std": "NaN"})
        else:
            summary.append({"metric": k, "mean": float(mval), "std": float(sval), "mean±std": f"{mval:.3f} ± {sval:.3f}"})

    df_summary = pd.DataFrame(summary)
    summary_path = os.path.join(cfg.save_dir, cfg.summary_table_name)
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig", float_format="%.9g")
    print(f"\n📌 Saved paper summary table (mean±std) -> {summary_path}")

    df_best_path = os.path.join(cfg.save_dir, "best_per_fold_metrics.csv")
    df_best.to_csv(df_best_path, index=False, encoding="utf-8-sig", float_format="%.9g")
    print(f"📌 Saved best-per-fold metrics -> {df_best_path}")

    print("\n✅ All folds finished.")
