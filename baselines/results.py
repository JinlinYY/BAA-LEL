from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd


def normalize_oof_frame(rows) -> pd.DataFrame:
    """Build a deterministically ordered OOF frame with canonical string PIDs."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    if "pid" not in frame.columns:
        raise ValueError("OOF results must include a pid column")
    frame["pid"] = frame["pid"].astype(str)
    return frame.sort_values("pid").reset_index(drop=True)


def save_per_class_metrics(path: Path, per_class: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Persist full-precision one-vs-rest classification metrics."""
    rows = [{"class": str(name), **{key: float(values[key]) for key in ["AUC", "Sensitivity", "Specificity"]}} for name, values in per_class.items()]
    frame = pd.DataFrame(rows, columns=["class", "AUC", "Sensitivity", "Specificity"])
    if frame.empty:
        raise ValueError("per_class metrics must not be empty")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame


class OOFClassificationWriter:
    def __init__(self, path: Path, class_names: Sequence[str]):
        self.path, self.class_names, self.rows = Path(path), list(class_names), []

    def add(self, fold: int, pids: Sequence[str], y_true: Sequence[int], y_proba: Sequence[Sequence[float]]):
        proba = np.asarray(y_proba, dtype=float)
        if proba.shape != (len(pids), len(self.class_names)) or len(y_true) != len(pids):
            raise ValueError("OOF arrays are not aligned")
        for i, pid in enumerate(pids):
            row = {"pid": str(pid), "fold": int(fold), "y_true": int(y_true[i]), "y_pred": int(proba[i].argmax())}
            row.update({f"proba_{name}": float(proba[i, c]) for c, name in enumerate(self.class_names)})
            self.rows.append(row)

    def save(self) -> pd.DataFrame:
        frame = normalize_oof_frame(self.rows)
        if frame.empty or frame["pid"].duplicated(keep=False).any():
            raise ValueError("each OOF pid must appear exactly once")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(self.path, index=False, encoding="utf-8-sig")
        np.savez(
            self.path.with_suffix(".npz"), pid=frame["pid"].astype(str).to_numpy(),
            fold=frame["fold"].to_numpy(), y_true=frame["y_true"].to_numpy(),
            y_proba=frame[[f"proba_{name}" for name in self.class_names]].to_numpy(),
        )
        return frame


class SegmentationCaseWriter:
    columns = ["pid", "fold", "Dice", "mIoU", "HD95", "ASSD"]

    def __init__(self, path: Path):
        self.path, self.rows = Path(path), []

    def add(self, fold: int, pid: str, metrics: Dict[str, float]):
        self.rows.append({"pid": str(pid), "fold": int(fold), **{k: float(metrics[k]) for k in self.columns[2:]}})

    def save(self) -> pd.DataFrame:
        frame = normalize_oof_frame(pd.DataFrame(self.rows, columns=self.columns))
        if frame.empty or frame["pid"].duplicated(keep=False).any():
            raise ValueError("each segmentation OOF pid must appear exactly once")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(self.path, index=False, encoding="utf-8-sig")
        return frame
