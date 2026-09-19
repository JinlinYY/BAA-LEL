"""Formal single-task baseline runner with nested model selection."""
from __future__ import annotations

import json
import hashlib
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from baselines.data import BaselineDataset, CaseRecord, ClinicalPreprocessor, load_clinical_frame
from baselines.metrics import classification_metrics, segmentation_case_metrics
from baselines.models import MODEL_METADATA, build_model
from baselines.models.segmentation import BRNLoss, SMUNetLoss
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: dict) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class BenchmarkConfig:
    dataset: str
    method: str
    task: str
    data_root: str = "data"
    output_root: str = "outputs/baselines"
    folds: int = 5
    seed: int = 42
    image_size: int = 256
    epochs: int = 50
    batch_size: int = 4
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 12
    inner_validation_fraction: float = .2
    threshold: float = .5
    medsam_checkpoint: str = "checkpoints/medsam_vit_b.pth"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    image_pretrained: bool = True
    resume: bool = False
    upstream_root: str = ""
    lr_candidates: tuple[float, ...] = ()
    effective_batch_size: int = 0
    max_folds: int = 0
    embedding_pretrain_epochs: int = 0
    miinet_finetune_blocks: int = 0
    backbone_lr: float = 1e-5
    miinet_warmup_epochs: int = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).flatten(1).sum(1)
    denominator = probability.flatten(1).sum(1) + target.flatten(1).sum(1)
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def _seg_logits(output):
    return output["logits"] if isinstance(output, dict) else output


def _classification_logits(output):
    return output["logits"] if isinstance(output, dict) else output


def _segmentation_forward(model: nn.Module, image: torch.Tensor, target: torch.Tensor, method: str):
    return model(image, oracle_mask=target) if method == "smu_net" else model(image)


def _segmentation_loss(method: str, output, target: torch.Tensor, bce: nn.Module) -> torch.Tensor:
    if method == "smu_net":
        return SMUNetLoss()(output, target)
    if method == "brn":
        return BRNLoss()(output, target)
    return .5 * bce(_seg_logits(output), target) + .5 * _dice_loss(_seg_logits(output), target)


def _classification_loss(output, target: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    loss = criterion(_classification_logits(output), target)
    return loss + output.get("aux_loss", loss.new_zeros(())) if isinstance(output, dict) else loss


class _FocalLoss(nn.Module):
    """Numerically stable form of the MIINet/KMNet author loss."""
    def __init__(self, gamma: float = 2.0, alpha: float = .25):
        super().__init__(); self.gamma, self.alpha = float(gamma), float(alpha)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probability = F.log_softmax(logits, dim=1)
        selected = log_probability.gather(1, target[:, None]).squeeze(1)
        probability = selected.exp()
        return (-self.alpha*(1-probability).pow(self.gamma)*selected).mean()


def _optimizer_and_scheduler(model: nn.Module, cfg: BenchmarkConfig):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if cfg.method == "miinet":
        if cfg.miinet_finetune_blocks:
            backbone = list(model.image_encoder.network.parameters())
            backbone_ids = {id(parameter) for parameter in backbone if parameter.requires_grad}
            head = [parameter for parameter in parameters if id(parameter) not in backbone_ids]
            optimizer = torch.optim.AdamW([
                {"params": [parameter for parameter in backbone if parameter.requires_grad], "lr": cfg.backbone_lr},
                {"params": head, "lr": cfg.lr},
            ], weight_decay=cfg.weight_decay)
        else:
            optimizer = torch.optim.Adam(parameters, lr=max(cfg.lr, 1e-3), weight_decay=cfg.weight_decay)
    elif cfg.method == "kmnet":


        optimizer = torch.optim.Adam(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.method == "hyperfusion":
        optimizer_class = torch.optim.Adam if cfg.upstream_root else torch.optim.AdamW
        optimizer = optimizer_class(parameters, lr=cfg.lr, weight_decay=1e-5)
    else:
        optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = None
    if cfg.method == "map":
        warmup = min(10, max(1, cfg.epochs//5))
        minimum_ratio = 1e-6/max(cfg.lr, 1e-12)
        def schedule(epoch):
            if epoch < warmup:
                return .1 + .9*(epoch+1)/warmup
            progress = (epoch-warmup)/max(1, cfg.epochs-warmup)
            return minimum_ratio+(1-minimum_ratio)*.5*(1+math.cos(math.pi*progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    return optimizer, scheduler


def _portable_model_state(model: nn.Module) -> dict:
    """Omit immutable released backbones instead of duplicating GBs per fold."""
    excluded = tuple(getattr(model, "checkpoint_exclude_prefixes", ()))
    return {name: value for name, value in model.state_dict().items() if not name.startswith(excluded)}


def _build_model(cfg: BenchmarkConfig, clinical_dim: int, num_classes: int, preprocessor: ClinicalPreprocessor | None = None) -> nn.Module:
    if cfg.task == "segmentation":
        kwargs: Dict[str, object] = {"checkpoint_path": cfg.medsam_checkpoint}
        if cfg.method in {"unet", "unetpp", "aau_net"}:
            kwargs["base_channels"] = 32
        elif cfg.method == "transunet":
            kwargs.update({"base_channels": 32, "transformer_dim": 256, "transformer_layers": 4})
        elif cfg.method == "smu_net":
            kwargs.update({
                "base_channels": 64, "saliency_backend": "lsc", "strict_lsc": True,
                "saliency_cache_dir": str(Path(cfg.output_root) / "cache" / "smu_lsc"),
            })
        elif cfg.method == "brn":
            kwargs.update({"base_channels": 64, "backbone": "resnet101"})
        return build_model(cfg.method, task="segmentation", **kwargs)
    kwargs = dict(
        clinical_dim=clinical_dim, num_classes=num_classes,
        pretrained=cfg.image_pretrained,
        clinical_feature_slices=getattr(preprocessor, "feature_slices_", None),
        clinical_feature_names=getattr(preprocessor, "feature_names_", None),
    )
    if cfg.upstream_root:
        from baselines.upstream_adapters import build_upstream_model
        model = build_upstream_model(cfg.method, Path(cfg.upstream_root), **kwargs)
        if cfg.method == "miinet" and cfg.miinet_finetune_blocks:
            model.image_encoder.enable_finetune(cfg.miinet_finetune_blocks, active=cfg.miinet_warmup_epochs <= 0)
            model.checkpoint_exclude_prefixes = ()
        return model
    return build_model(
        cfg.method, task="classification", clinical_dim=clinical_dim, num_classes=num_classes,
        checkpoint_path=cfg.medsam_checkpoint, pretrained=cfg.image_pretrained,
        clinical_feature_slices=getattr(preprocessor, "feature_slices_", None),
        clinical_feature_names=getattr(preprocessor, "feature_names_", None),
    )


def _dataset(records: Sequence[CaseRecord], clinical: np.ndarray, cfg: BenchmarkConfig) -> BaselineDataset:
    return BaselineDataset(records, clinical, cfg.image_size, augment=False, grayscale=cfg.dataset.lower() != "imaplusplus")


def _loader(dataset: BaselineDataset, cfg: BenchmarkConfig, shuffle: bool) -> DataLoader:
    worker_args = {"persistent_workers": True, "prefetch_factor": 8} if cfg.num_workers > 0 else {}
    return DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers,
        pin_memory=(torch.device(cfg.device).type == "cuda"), drop_last=bool(shuffle),
        **worker_args,
    )


def _criterion(records: Sequence[CaseRecord], num_classes: int, device: torch.device, method: str = "", upstream: bool = False) -> nn.Module:
    if method in {"miinet", "kmnet"} or (upstream and method == "amf_medit"):
        return _FocalLoss(gamma=2, alpha=.25)
    if upstream and method == "map":
        return nn.CrossEntropyLoss()
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights /= weights.mean()
    smoothing = 0.0 if upstream and method == "hyperfusion" else .05
    return nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float32, device=device), label_smoothing=smoothing)


def _train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, cfg: BenchmarkConfig, device: torch.device) -> float:
    model.train()


    amp_enabled = cfg.amp and device.type == "cuda" and cfg.method not in {"kmnet", "amf_medit"}
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    bce = nn.BCEWithLogitsLoss()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    accumulation = min(max(1, int(cfg.grad_accum_steps)), max(1, len(loader)))
    for step, batch in enumerate(loader, 1):
        image = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            if cfg.task == "segmentation":
                target = batch["mask"].to(device, non_blocking=True)
                output = _segmentation_forward(model, image, target, cfg.method)
                loss = _segmentation_loss(cfg.method, output, target, bce)
            else:
                target = batch["label"].to(device, non_blocking=True)
                output = model(image, batch["clinical"].to(device, non_blocking=True))
                loss = _classification_loss(output, target, criterion)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite training loss for {cfg.method}; refusing to write an invalid fold"
            )
        window_start = ((step - 1) // accumulation) * accumulation
        first_sample = window_start * loader.batch_size
        window_samples = min(accumulation * loader.batch_size, len(loader.dataset) - first_sample)
        sample_weight = image.shape[0] / max(1, window_samples)
        scaler.scale(loss * sample_weight).backward()
        losses.append(float(loss.detach().cpu()))
        if step % accumulation == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
    return float(np.mean(losses))


@torch.no_grad()
def _evaluate_segmentation(
    model: nn.Module,
    loader: DataLoader,
    cfg: BenchmarkConfig,
    device: torch.device,
    fold: int | None = None,
    *,
    full_metrics: bool = True,
):
    model.eval()
    rows = []
    for batch in loader:
        image, target = batch["image"].to(device), batch["mask"].to(device)
        prediction = (torch.sigmoid(_seg_logits(_segmentation_forward(model, image, target, cfg.method))) >= cfg.threshold).cpu().numpy()
        for pid, predicted_mask, true_mask in zip(batch["pid"], prediction, target.cpu().numpy()):
            if full_metrics:
                metrics = segmentation_case_metrics(predicted_mask, true_mask)
            else:
                predicted = predicted_mask.astype(bool, copy=False)
                truth = true_mask.astype(bool, copy=False)
                intersection = np.logical_and(predicted, truth).sum(dtype=np.float64)
                denominator = predicted.sum(dtype=np.float64) + truth.sum(dtype=np.float64)
                dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
                metrics = {"Dice": float(dice)}
            row = {"pid": str(pid), **metrics}
            if fold is not None:
                row["fold"] = int(fold)
            rows.append(row)
    return rows


@torch.no_grad()
def _evaluate_classification(model: nn.Module, loader: DataLoader, device: torch.device, fold: int | None = None):
    model.eval()
    rows = []
    for batch in loader:
        probabilities = torch.softmax(_classification_logits(model(batch["image"].to(device), batch["clinical"].to(device))), 1).cpu().numpy()
        for pid, label, probability in zip(batch["pid"], batch["label"].numpy(), probabilities):
            row = {"pid": str(pid), "y_true": int(label), "y_pred": int(probability.argmax())}
            row.update({f"proba_{index}": float(value) for index, value in enumerate(probability)})
            if fold is not None:
                row["fold"] = int(fold)
            rows.append(row)
    return rows


def _score(task: str, rows, names: Sequence[str]) -> float:
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("development evaluation returned no cases")
    if task == "segmentation":
        return float(frame["Dice"].mean())
    probabilities = frame[[f"proba_{name}" for name in names]].to_numpy()
    if not np.isfinite(probabilities).all():
        return -np.inf
    return float(classification_metrics(frame["y_true"], probabilities, names)["Macro-AUC"])


def _clinical_arrays(frame: pd.DataFrame, preprocessor: ClinicalPreprocessor, record_sets: Sequence[Sequence[CaseRecord]]) -> list[np.ndarray]:
    return [preprocessor.transform(frame.loc[[record.pid for record in records]]) for records in record_sets]


def _prepare_upstream_model(model: nn.Module, cfg: BenchmarkConfig, clinical: np.ndarray, records: Sequence[CaseRecord], device: torch.device) -> None:
    if not cfg.upstream_root or cfg.method != "hyperfusion":
        return
    values = torch.as_tensor(clinical, dtype=torch.float32, device=device)
    labels = torch.as_tensor([record.label for record in records], dtype=torch.long, device=device)
    embedding = model.clinical_embedding
    head = nn.Linear(8, int(labels.max().item()) + 1).to(device)
    optimizer = torch.optim.Adam([*embedding.parameters(), *head.parameters()], lr=1e-4, weight_decay=1e-5)
    embedding.train(); head.train()
    for _ in range(max(1, int(cfg.embedding_pretrain_epochs))):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(embedding(values)), labels)
        loss.backward(); optimizer.step()
    model.hyper_residual_block.downsample.initialize_input_variance(values)


def _fit_selected_epoch(cfg: BenchmarkConfig, train_records, dev_records, clinical_frame, num_classes, names, device, fold_dir: Path, fold: int):
    preprocessor = ClinicalPreprocessor().fit(clinical_frame.loc[[record.pid for record in train_records]])
    train_clinical, dev_clinical = _clinical_arrays(clinical_frame, preprocessor, [train_records, dev_records])
    train_loader = _loader(_dataset(train_records, train_clinical, cfg), cfg, True)
    dev_loader = _loader(_dataset(dev_records, dev_clinical, cfg), cfg, False)
    best_epoch, best_score, best_lr, history = 1, -np.inf, cfg.lr, []
    for candidate_lr in (cfg.lr_candidates or (cfg.lr,)):
        candidate_cfg = replace(cfg, lr=float(candidate_lr))
        seed_everything(cfg.seed + fold)
        model = _build_model(candidate_cfg, train_clinical.shape[1], num_classes, preprocessor).to(device)
        _prepare_upstream_model(model, candidate_cfg, train_clinical, train_records, device)
        optimizer, scheduler = _optimizer_and_scheduler(model, candidate_cfg)
        criterion = nn.BCEWithLogitsLoss() if cfg.task == "segmentation" else _criterion(train_records, num_classes, device, cfg.method, bool(cfg.upstream_root))
        candidate_best, stale = -np.inf, 0
        for epoch in range(1, cfg.epochs + 1):
            if cfg.method == "miinet" and cfg.miinet_finetune_blocks and epoch == cfg.miinet_warmup_epochs + 1:
                model.image_encoder.set_finetune_active(True)
                stale = 0
            loss = _train_epoch(model, train_loader, optimizer, criterion, candidate_cfg, device)
            if scheduler is not None:
                scheduler.step()


            rows = _evaluate_segmentation(model, dev_loader, candidate_cfg, device, full_metrics=False) if cfg.task == "segmentation" else _evaluate_classification(model, dev_loader, device)
            score = _score(cfg.task, rows, names)
            history.append({"fold": fold, "learning_rate": candidate_lr, "epoch": epoch, "train_loss": loss, "development_score": score})
            if score > candidate_best:
                candidate_best, stale = score, 0
            else:
                stale += 1
            if score > best_score:
                best_epoch, best_score, best_lr = epoch, score, float(candidate_lr)
            if stale >= cfg.patience and epoch > cfg.miinet_warmup_epochs:
                break
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pd.DataFrame(history).to_csv(fold_dir / "selection_metrics.csv", index=False, encoding="utf-8-sig")
    return int(best_epoch), float(best_score), float(best_lr)


def _train_final_and_predict(cfg: BenchmarkConfig, train_records, test_records, clinical_frame, num_classes, selected_epoch, selected_lr, device, fold_dir: Path, fold: int, run_identity: dict):
    cfg = replace(cfg, lr=float(selected_lr))
    preprocessor = ClinicalPreprocessor().fit(clinical_frame.loc[[record.pid for record in train_records]])
    train_clinical, test_clinical = _clinical_arrays(clinical_frame, preprocessor, [train_records, test_records])
    train_loader = _loader(_dataset(train_records, train_clinical, cfg), cfg, True)
    test_loader = _loader(_dataset(test_records, test_clinical, cfg), cfg, False)
    seed_everything(cfg.seed + fold)
    model = _build_model(cfg, train_clinical.shape[1], num_classes, preprocessor).to(device)
    _prepare_upstream_model(model, cfg, train_clinical, train_records, device)
    optimizer, scheduler = _optimizer_and_scheduler(model, cfg)
    criterion = nn.BCEWithLogitsLoss() if cfg.task == "segmentation" else _criterion(train_records, num_classes, device, cfg.method, bool(cfg.upstream_root))
    for epoch in range(1, selected_epoch + 1):
        if cfg.method == "miinet" and cfg.miinet_finetune_blocks and epoch == cfg.miinet_warmup_epochs + 1:
            model.image_encoder.set_finetune_active(True)
        _train_epoch(model, train_loader, optimizer, criterion, cfg, device)
        if scheduler is not None:
            scheduler.step()
    torch.save({
        "model_state": _portable_model_state(model), "selected_epoch": selected_epoch, "selected_lr": selected_lr,
        "fold": fold, "method": cfg.method, "task": cfg.task,
        "clinical_feature_dim": int(train_clinical.shape[1]),
        "clinical_preprocessor": preprocessor.state_dict(),
        "excluded_released_backbone_prefixes": list(getattr(model, "checkpoint_exclude_prefixes", ())),
        "run_identity": run_identity,
    }, fold_dir / "model.pt")
    return _evaluate_segmentation(model, test_loader, cfg, device, fold) if cfg.task == "segmentation" else _evaluate_classification(model, test_loader, device, fold)


def run_benchmark(cfg: BenchmarkConfig):
    metadata = MODEL_METADATA.get(cfg.method)
    if cfg.task not in {"segmentation", "classification"}:
        raise ValueError("single-task runner accepts segmentation or classification only")
    if metadata is None or metadata.get("task") != cfg.task:
        raise ValueError(f"method {cfg.method!r} is not a registered {cfg.task} baseline")
    if cfg.method in {"hetmed", "baa_lel", "mtanet"}:
        raise ValueError(f"method {cfg.method!r} requires a dedicated runner")
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    records = protocol_records(cfg.dataset, cfg.data_root)
    counts = np.bincount(np.asarray([record.label for record in records], dtype=np.int64))
    if counts[counts > 0].min() < cfg.folds:
        raise ValueError(f"each class needs at least {cfg.folds} cases; counts={counts.tolist()}")
    outer = create_or_load_patient_manifest(cfg.output_root, cfg.dataset, records, folds=cfg.folds, seed=cfg.seed)
    inner = create_or_load_inner_manifest(cfg.output_root, cfg.dataset, records, outer, seed=cfg.seed, validation_fraction=cfg.inner_validation_fraction)
    validate_shared_segmentation_folds(outer, records)
    if cfg.upstream_root:
        from baselines.upstream_protocol import load_upstream_clinical_frame
        clinical_frame = load_upstream_clinical_frame(cfg.dataset, Path(cfg.data_root), records)
    else:
        clinical_frame = load_clinical_frame(cfg.dataset, Path(cfg.data_root), records)
    if cfg.task == "classification" and clinical_frame.shape[1] == 0:
        raise ValueError(f"{cfg.method} requires clinical features")
    names, root = class_names(records), Path(cfg.output_root) / cfg.dataset / cfg.task / cfg.method
    root.mkdir(parents=True, exist_ok=True)
    provenance = None
    adapter = None
    if cfg.upstream_root:
        from baselines.upstream import UpstreamSourceRegistry
        from baselines.upstream_adapters import adapter_provenance
        provenance = UpstreamSourceRegistry.default().verify_one(cfg.method, Path(cfg.upstream_root))
        adapter = adapter_provenance()
    dataset_dir = {"HER2USC": "HER2", "LMNUSC": "LNM", "BrEaST": "BrEaST"}.get(cfg.dataset, cfg.dataset)
    clinical_path = (Path(cfg.data_root) / dataset_dir / "metadata" / "clinical_binary.csv") if cfg.dataset == "IMAplusplus" else (Path(cfg.data_root) / dataset_dir / "clinical.xlsx")
    clinical_hash = _sha256(clinical_path) if clinical_path.is_file() else None
    identity_config = asdict(cfg)
    identity_config["resume"] = False
    run_identity = {
        "config_hash": _canonical_hash({
            "config": identity_config, "upstream": provenance, "adapter": adapter,
            "clinical_data_sha256": clinical_hash,
            "clinical_columns": list(map(str, clinical_frame.columns)),
            "fold_fingerprint": outer["split_fingerprint"],
            "inner_fold_fingerprint": inner["split_fingerprint"],
        }),
        "upstream": provenance,
        "adapter": adapter,
        "clinical_data_sha256": clinical_hash,
        "clinical_columns": list(map(str, clinical_frame.columns)),
        "fold_fingerprint": outer["split_fingerprint"],
        "inner_fold_fingerprint": inner["split_fingerprint"],
    }
    manifest_path = root / "run_manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("config_hash") != run_identity["config_hash"]:
            raise RuntimeError("resume refused: run configuration/source/data/fold identity changed")
    else:
        manifest_path.write_text(json.dumps(run_identity, ensure_ascii=False, indent=2), encoding="utf-8")
    inner_by_fold = {int(entry["fold"]): entry for entry in inner["folds"]}
    rows_all, fold_rows = [], []
    selected_outer_folds = outer["folds"][:cfg.max_folds] if cfg.max_folds > 0 else outer["folds"]
    for outer_fold in selected_outer_folds:
        fold = int(outer_fold["fold"])
        fold_dir = root / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        output_name = "oof_segmentation_cases.csv" if cfg.task == "segmentation" else "oof_classification.csv"
        existing = fold_dir / output_name
        if cfg.resume and existing.exists():
            saved_selection = json.loads((fold_dir / "selection.json").read_text(encoding="utf-8"))
            if saved_selection.get("config_hash") != run_identity["config_hash"]:
                raise RuntimeError(f"resume refused for fold {fold}: identity mismatch")
            rows_all.extend(pd.read_csv(existing, dtype={"pid": str}).to_dict("records"))
            fold_rows.append(saved_selection)
            continue
        entry = inner_by_fold[fold]
        require_mask = cfg.task == "segmentation"
        train_records = records_for_pids(records, outer_fold["train_pids"], require_mask=require_mask)
        test_records = records_for_pids(records, outer_fold["val_pids"], require_mask=require_mask)
        dev_train_records = records_for_pids(records, entry["train_pids"], require_mask=require_mask)
        dev_records = records_for_pids(records, entry["dev_pids"], require_mask=require_mask)
        if not all((train_records, test_records, dev_train_records, dev_records)):
            raise RuntimeError(f"fold {fold} has no usable {cfg.task} cases")
        selection_path = fold_dir / "selection.json"
        if cfg.resume and selection_path.exists():
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            if selection.get("config_hash") != run_identity["config_hash"]:
                raise RuntimeError(f"resume refused for fold {fold}: selection identity mismatch")
            selected_epoch, selection_score = int(selection["selected_epoch"]), float(selection["selection_score"])
            selected_lr = float(selection.get("selected_lr", cfg.lr))
        else:
            selected_epoch, selection_score, selected_lr = _fit_selected_epoch(cfg, dev_train_records, dev_records, clinical_frame, len(names), names, device, fold_dir, fold)
            selection = {"fold": fold, "selected_epoch": selected_epoch, "selected_lr": selected_lr, "selection_score": selection_score, "config_hash": run_identity["config_hash"]}
            selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
        fold_rows.append(selection)
        rows = _train_final_and_predict(cfg, train_records, test_records, clinical_frame, len(names), selected_epoch, selected_lr, device, fold_dir, fold, run_identity)
        pd.DataFrame(rows).to_csv(existing, index=False, encoding="utf-8-sig")
        rows_all.extend(rows)
    oof = normalize_oof_frame(rows_all)
    if oof.empty or oof["pid"].duplicated().any():
        raise RuntimeError("OOF results must contain each evaluated patient exactly once")
    if cfg.task == "segmentation":
        selected_test_pids = {str(pid) for item in selected_outer_folds for pid in item["val_pids"]}
        expected = {record.pid for record in records if record.mask_paths and record.pid in selected_test_pids}
        if set(oof["pid"].astype(str)) != expected:
            raise RuntimeError("segmentation OOF does not match the mask-bearing cohort")
        oof.to_csv(root / "oof_segmentation_cases.csv", index=False, encoding="utf-8-sig")
        flat = {metric: float(oof[metric].mean()) for metric in ("Dice", "mIoU", "HD95", "ASSD")}
        fold_metrics = oof.groupby("fold")[["Dice", "mIoU", "HD95", "ASSD"]].mean().reset_index()
    else:
        selected_test_pids = {str(pid) for item in selected_outer_folds for pid in item["val_pids"]}
        if set(oof["pid"].astype(str)) != selected_test_pids:
            raise RuntimeError("classification OOF does not match the formal cohort")
        probability_columns = [f"proba_{name}" for name in names]
        if not np.allclose(oof[probability_columns].sum(1), 1.0, atol=1e-4):
            raise RuntimeError("classification probabilities must sum to one")
        oof.to_csv(root / "oof_classification.csv", index=False, encoding="utf-8-sig")
        overall = classification_metrics(oof["y_true"], oof[probability_columns].to_numpy(), names)
        save_per_class_metrics(root / "per_class_metrics.csv", overall["per_class"])
        flat = {key: value for key, value in overall.items() if key != "per_class"}
        for name, values in overall["per_class"].items():
            flat.update({f"class_{name}_{key}": value for key, value in values.items()})
        fold_metrics = pd.DataFrame([
            {"fold": int(fold), **{key: value for key, value in classification_metrics(group["y_true"], group[probability_columns].to_numpy(), names).items() if key != "per_class"}}
            for fold, group in oof.groupby("fold")
        ])
    pd.DataFrame(fold_rows).to_csv(root / "selection_per_fold.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(root / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    metric_columns = [column for column in fold_metrics.columns if column != "fold"]
    fold_summary = {
        column: {"mean": float(fold_metrics[column].mean()), "sd": float(fold_metrics[column].std(ddof=1))}
        for column in metric_columns
    }
    (root / "fold_mean_sd.json").write_text(json.dumps(fold_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "dataset": cfg.dataset, "task": cfg.task, "method": cfg.method, "display_name": metadata["display_name"],
        "implementation": metadata["implementation"], "source": metadata["source"], "version": metadata["version"],
        "upstream_version": metadata["upstream_version"], "interactive": metadata["interactive"],
        "eligible_for_automatic_comparison": metadata["eligible_for_automatic_comparison"], "protocol_version": PROTOCOL_VERSION,
        "fold_fingerprint": outer["split_fingerprint"], "inner_fold_fingerprint": inner["split_fingerprint"],
        "evaluated_folds": [int(item["fold"]) for item in selected_outer_folds],
        **{key: round(float(value), 3) for key, value in flat.items()},
    }
    if cfg.upstream_root:
        implementation_label = {
            "miinet": "audited_author_module_adapter",
            "kmnet": "audited_author_module_adapter",
            "hyperfusion": "2d_official_code_derived_adaptation",
            "amf_medit": "audited_author_module_adapter",
            "map": "us_clinical_official_code_adaptation",
        }[cfg.method]
        summary.update({
            "implementation": implementation_label,
            "upstream": provenance,
            "adapter": adapter,
            "clinical_columns": list(map(str, clinical_frame.columns)),
        })
        summary["clinical_data_sha256"] = clinical_hash
        summary["config_hash"] = run_identity["config_hash"]
        summary["fold_mean_sd"] = fold_summary
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "run_config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
