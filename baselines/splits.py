import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def _canonical_records(pids: Sequence[str], labels: Sequence[int]) -> List[Tuple[str, int]]:
    if len(pids) != len(labels) or len(set(map(str, pids))) != len(pids):
        raise ValueError("pids and labels must align and pids must be unique")
    return sorted([(str(pid), int(label)) for pid, label in zip(pids, labels)])


def _input_order_records(pids: Sequence[str], labels: Sequence[int]) -> List[Tuple[str, int]]:
    if len(pids) != len(labels) or len(set(map(str, pids))) != len(pids):
        raise ValueError("pids and labels must align and pids must be unique")
    return [(str(pid), int(label)) for pid, label in zip(pids, labels)]


def _fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _split_fingerprint(folds: Sequence[Dict[str, object]]) -> str:
    return _fingerprint(folds)


def _validate_fold_coverage(
    folds: Sequence[Dict[str, object]],
    expected_pids: Sequence[str],
    n_splits: int,
) -> None:
    if len(folds) != int(n_splits):
        raise ValueError(f"fold manifest has {len(folds)} folds, expected {n_splits}")
    expected = set(map(str, expected_pids))
    observed: List[str] = []
    for expected_fold, fold in enumerate(folds, 1):
        if int(fold.get("fold", -1)) != expected_fold:
            raise ValueError("fold manifest fold identifiers must be contiguous from 1")
        train = set(map(str, fold.get("train_pids", [])))
        val = list(map(str, fold.get("val_pids", [])))
        if train & set(val):
            raise ValueError(f"fold {expected_fold} has overlapping train and validation patients")
        if train | set(val) != expected:
            raise ValueError(f"fold {expected_fold} does not cover the expected cohort")
        observed.extend(val)
    if len(observed) != len(expected) or set(observed) != expected or len(set(observed)) != len(observed):
        raise ValueError("every patient must appear in exactly one validation fold")


def create_or_load_folds(
    path: Path, pids: Sequence[str], labels: Sequence[int], n_splits: int = 5, seed: int = 42
) -> Dict[str, object]:
    records = _canonical_records(pids, labels)
    fingerprint = _fingerprint(records)
    path = Path(path)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected = (fingerprint, int(n_splits), int(seed))
        actual = (manifest.get("dataset_fingerprint"), manifest.get("n_splits"), manifest.get("seed"))
        if actual != expected:
            raise ValueError("existing fold manifest does not match this dataset/configuration")
        if "split_fingerprint" not in manifest:
            manifest["split_fingerprint"] = _split_fingerprint(manifest["folds"])
            path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        _validate_fold_coverage(manifest["folds"], [record[0] for record in records], n_splits)
        return manifest

    sorted_pids = np.asarray([r[0] for r in records])
    sorted_labels = np.asarray([r[1] for r in records])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(sorted_pids, sorted_labels), 1):
        folds.append({
            "fold": fold,
            "train_pids": sorted_pids[train_idx].tolist(),
            "val_pids": sorted_pids[val_idx].tolist(),
        })
    manifest = {
        "version": 1, "n_splits": int(n_splits), "seed": int(seed),
        "dataset_fingerprint": fingerprint, "folds": folds,
    }
    manifest["split_fingerprint"] = _split_fingerprint(folds)
    _validate_fold_coverage(folds, sorted_pids.tolist(), n_splits)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def create_or_load_group_folds(
    path: Path,
    pids: Sequence[str],
    labels: Sequence[int],
    groups: Sequence[str],
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, object]:
    """Create leakage-safe stratified folds with each group held in one fold."""
    if len(pids) != len(groups):
        raise ValueError("pids and groups must align")
    group_by_pid = {str(pid): str(group) for pid, group in zip(pids, groups)}
    records = _canonical_records(pids, labels)
    sorted_pids = np.asarray([record[0] for record in records])
    sorted_labels = np.asarray([record[1] for record in records])
    sorted_groups = np.asarray([group_by_pid[pid] for pid in sorted_pids])
    fingerprint = _fingerprint(records)
    group_fingerprint = _fingerprint(sorted(zip(sorted_pids.tolist(), sorted_groups.tolist())))
    path = Path(path)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected = (fingerprint, group_fingerprint, int(n_splits), int(seed))
        actual = (
            manifest.get("dataset_fingerprint"),
            manifest.get("group_fingerprint"),
            manifest.get("n_splits"),
            manifest.get("seed"),
        )
        if actual != expected:
            raise ValueError("existing group fold manifest does not match this dataset/configuration")
        _validate_fold_coverage(manifest["folds"], sorted_pids.tolist(), n_splits)
        return manifest

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(sorted_pids, sorted_labels, groups=sorted_groups), 1
    ):
        train_groups = set(sorted_groups[train_idx])
        val_groups = set(sorted_groups[val_idx])
        if train_groups & val_groups:
            raise RuntimeError(f"group leakage detected in fold {fold}")
        folds.append({
            "fold": fold,
            "train_pids": sorted_pids[train_idx].tolist(),
            "val_pids": sorted_pids[val_idx].tolist(),
        })
    manifest = {
        "version": 1,
        "ordering_policy": "stratified_group",
        "n_splits": int(n_splits),
        "seed": int(seed),
        "dataset_fingerprint": fingerprint,
        "group_fingerprint": group_fingerprint,
        "folds": folds,
    }
    manifest["split_fingerprint"] = _split_fingerprint(folds)
    _validate_fold_coverage(folds, sorted_pids.tolist(), n_splits)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def create_or_load_input_order_folds(
    path: Path,
    pids: Sequence[str],
    labels: Sequence[int],
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, object]:
    """Persist the pre-manifest StratifiedKFold split using the caller's raw PID order.

    The initial submission code passed the `os.listdir`-derived order from
    ``build_pid_and_labels`` directly to ``StratifiedKFold``.  This is
    deliberately distinct from ``create_or_load_folds``, which canonicalizes
    PIDs for comparative experiments.  The ordered PID list and its hash are
    stored so a different directory enumeration cannot silently reuse a split.
    """
    records = _input_order_records(pids, labels)
    ordered_pids = [record[0] for record in records]
    ordered_labels = [record[1] for record in records]
    dataset_fingerprint = _fingerprint(sorted(records))
    input_order_fingerprint = _fingerprint(records)
    path = Path(path)

    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected = (
            "input_order",
            dataset_fingerprint,
            input_order_fingerprint,
            int(n_splits),
            int(seed),
        )
        actual = (
            manifest.get("ordering_policy"),
            manifest.get("dataset_fingerprint"),
            manifest.get("input_order_fingerprint"),
            manifest.get("n_splits"),
            manifest.get("seed"),
        )
        if actual != expected:
            if actual[0] == "input_order" and actual[1] == dataset_fingerprint:
                raise ValueError("existing reference fold manifest does not match this input ordering")
            raise ValueError("existing reference fold manifest does not match this dataset/configuration")
        if manifest.get("input_order_pids") != ordered_pids:
            raise ValueError("existing reference fold manifest does not match this input ordering")
        if "split_fingerprint" not in manifest:
            manifest["split_fingerprint"] = _split_fingerprint(manifest["folds"])
            path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        _validate_fold_coverage(manifest["folds"], ordered_pids, n_splits)
        return manifest

    pid_array = np.asarray(ordered_pids)
    label_array = np.asarray(ordered_labels)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(pid_array, label_array), 1):
        folds.append({
            "fold": fold,
            "train_pids": pid_array[train_idx].tolist(),
            "val_pids": pid_array[val_idx].tolist(),
        })
    _validate_fold_coverage(folds, ordered_pids, n_splits)
    manifest = {
        "version": 2,
        "ordering_policy": "input_order",
        "n_splits": int(n_splits),
        "seed": int(seed),
        "dataset_fingerprint": dataset_fingerprint,
        "input_order_fingerprint": input_order_fingerprint,
        "input_order_pids": ordered_pids,
        "folds": folds,
        "split_fingerprint": _split_fingerprint(folds),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
