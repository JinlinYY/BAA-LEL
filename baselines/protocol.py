"""Protocol."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

from baselines.data import CaseRecord, load_records
from baselines.splits import create_or_load_folds, create_or_load_group_folds


PROTOCOL_VERSION = "patient-cv-v1"


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def protocol_records(
    dataset: str,
    data_root: str | Path,
    clinical_name: str | Path = "clinical.xlsx",
) -> list[CaseRecord]:
    """Return records under the task labels fixed by the formal protocol.

    BrEaST contains only four ``normal`` cases, so its source three-class
    labels cannot support stratified five-fold classification.  The protocol
    therefore reports the prespecified non-malignant (benign + normal) versus
    malignant task while keeping every available patient.
    """
    records = load_records(dataset, Path(data_root), clinical_name=clinical_name)
    if dataset.lower() != "breast":
        return records
    return [replace(record, label=int(record.label == 1)) for record in records]


def class_names(records: Sequence[CaseRecord]) -> list[str]:
    return [str(label) for label in sorted({int(record.label) for record in records})]


def create_or_load_patient_manifest(
    output_root: str | Path,
    dataset: str,
    records: Sequence[CaseRecord],
    *,
    folds: int = 5,
    seed: int = 42,
) -> Dict[str, object]:
    """Persist the sole outer split used by all tasks on a dataset."""
    root = Path(output_root)
    groups = [record.group_id for record in records]
    grouped = any(group is not None for group in groups)
    if grouped and not all(group is not None for group in groups):
        raise ValueError("group identifiers must be present for every record or none")
    path = root / "splits" / dataset / f"patient_seed{seed}_{folds}fold.json"
    if grouped:
        manifest = create_or_load_group_folds(
            path, [record.pid for record in records], [record.label for record in records],
            [str(group) for group in groups], folds, seed,
        )
    else:
        manifest = create_or_load_folds(
            path, [record.pid for record in records], [record.label for record in records], folds, seed,
        )
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        manifest = dict(manifest)
        manifest["protocol_version"] = PROTOCOL_VERSION
        (root / "splits" / dataset).mkdir(parents=True, exist_ok=True)
        (root / "splits" / dataset / f"patient_seed{seed}_{folds}fold.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return manifest


def create_or_load_inner_manifest(
    output_root: str | Path,
    dataset: str,
    records: Sequence[CaseRecord],
    outer_manifest: Dict[str, object],
    *,
    seed: int = 42,
    validation_fraction: float = 0.2,
) -> Dict[str, object]:
    """Create deterministic development splits strictly inside every outer train set."""
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, .5)")
    by_pid = {record.pid: record for record in records}
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "outer_split_fingerprint": outer_manifest["split_fingerprint"],
        "seed": int(seed),
        "validation_fraction": float(validation_fraction),
        "folds": [],
    }
    for outer in outer_manifest["folds"]:
        fold = int(outer["fold"])
        train_pids = np.asarray(list(map(str, outer["train_pids"])))
        labels = np.asarray([by_pid[pid].label for pid in train_pids], dtype=np.int64)
        counts = np.bincount(labels)
        if counts[counts > 0].min() < 2:
            raise ValueError(f"outer fold {fold} cannot make a stratified inner split; counts={counts.tolist()}")
        train_groups = np.asarray([by_pid[pid].group_id for pid in train_pids], dtype=object)
        if all(group is not None for group in train_groups):
            splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + fold)
            inner_train, inner_val = next(splitter.split(train_pids, labels, groups=train_groups))
            if set(train_groups[inner_train]) & set(train_groups[inner_val]):
                raise RuntimeError(f"inner group leakage detected in outer fold {fold}")
        else:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed + fold)
            inner_train, inner_val = next(splitter.split(train_pids, labels))
        payload["folds"].append({
            "fold": fold,
            "train_pids": sorted(train_pids[inner_train].tolist()),
            "dev_pids": sorted(train_pids[inner_val].tolist()),
            "outer_test_pids": sorted(map(str, outer["val_pids"])),
        })
    payload["split_fingerprint"] = _fingerprint(payload["folds"])
    path = Path(output_root) / "splits" / dataset / f"inner_seed{seed}_dev{validation_fraction:.2f}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        expected = (payload["outer_split_fingerprint"], payload["seed"], payload["validation_fraction"], payload["split_fingerprint"])
        actual = (existing.get("outer_split_fingerprint"), existing.get("seed"), existing.get("validation_fraction"), existing.get("split_fingerprint"))
        if actual != expected:
            raise ValueError("existing inner split does not match the formal patient protocol")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def records_for_pids(records: Sequence[CaseRecord], pids: Iterable[str], *, require_mask: bool = False) -> list[CaseRecord]:
    by_pid = {record.pid: record for record in records}
    resolved = [by_pid[str(pid)] for pid in pids]
    return [record for record in resolved if record.mask_paths] if require_mask else resolved


def validate_shared_segmentation_folds(
    outer_manifest: Dict[str, object], records: Sequence[CaseRecord]
) -> None:
    """Assert every mask-bearing case remains assigned to its outer patient fold."""
    expected = {record.pid for record in records if record.mask_paths}
    observed: set[str] = set()
    for fold in outer_manifest["folds"]:
        observed.update(str(pid) for pid in fold["val_pids"] if str(pid) in expected)
    if observed != expected:
        raise ValueError("mask-bearing patients are not fully covered by shared outer folds")
