import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class CaseRecord:
    pid: str
    image_path: Path
    mask_paths: tuple
    label: int
    group_id: Optional[str] = None


def _flat_records(root: Path, clinical_name="clinical.xlsx") -> List[CaseRecord]:
    frame = pd.read_excel(root / clinical_name, engine="openpyxl")
    pid_col, label_col = frame.columns[0], frame.columns[-1]
    labels = _convert_labels(frame[label_col])
    pid_to_label = {str(pid).strip(): int(label) for pid, label in zip(frame[pid_col], labels)}
    images = {p.stem: p for p in (root / "images").iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    masks = {p.stem: p for p in (root / "masks").iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    return [CaseRecord(pid, image, (masks[pid],) if pid in masks else (), pid_to_label[pid]) for pid, image in sorted(images.items()) if pid in pid_to_label]


def _convert_labels(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(int).to_numpy()
    normalized = series.astype(str).str.strip().str.upper()
    mapping = {"LN0": 0, "LN1-3": 1, "LN1–3": 1, "LN4+": 2, "BENIGN": 0, "MALIGNANT": 1, "NORMAL": 2}
    mapped = normalized.map(mapping)
    if mapped.isna().any():
        raise ValueError(f"unsupported labels: {normalized[mapped.isna()].unique().tolist()}")
    return mapped.astype(int).to_numpy()


def _breast_records(
    root: Path, clinical_name: Union[str, Path] = "clinical.xlsx"
) -> List[CaseRecord]:
    frame = pd.read_excel(root / clinical_name, engine="openpyxl")
    folder = root / "images and masks"
    labels = _convert_labels(frame["Classification"])
    records = []
    for (_, row), label in zip(frame.iterrows(), labels):
        image = folder / str(row["Image_filename"])
        mask = folder / str(row["Mask_tumor_filename"])
        if image.exists():
            records.append(CaseRecord(str(row["CaseID"]).strip(), image, (mask,) if mask.exists() else (), int(label)))
    return records


def _busi_records(root: Path) -> List[CaseRecord]:
    class_map = {"benign": 0, "malignant": 1, "normal": 2}
    records = []
    for class_name, label in class_map.items():
        folder = root / class_name
        for image in sorted(folder.glob("*")):
            if image.suffix.lower() not in IMAGE_EXTENSIONS or re.search(r"_mask(?:_\d+)?$", image.stem):
                continue
            masks = tuple(sorted(folder.glob(image.stem + "_mask*" + image.suffix)))
            records.append(CaseRecord(f"{class_name}/{image.stem}", image, masks, label))
    return records


def _isic_records(root: Path) -> List[CaseRecord]:
    image_dirs = [root / "images", root / "ISIC2018_Task1-2_Training_Input"]
    mask_dirs = [root / "masks", root / "ISIC2018_Task1_Training_GroundTruth"]
    image_dir = next((p for p in image_dirs if p.exists()), None)
    mask_dir = next((p for p in mask_dirs if p.exists()), None)
    if image_dir is None or mask_dir is None:
        raise FileNotFoundError(
            f"ISIC2018 not found under {root}. Expected images/masks or official Task1 training folders."
        )
    records = []
    for image in sorted(image_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        candidates = [mask_dir / f"{image.stem}_segmentation.png", mask_dir / f"{image.stem}.png"]
        mask = next((p for p in candidates if p.exists()), None)
        if mask:
            records.append(CaseRecord(image.stem, image, (mask,), 0))
    return records


def _imaplusplus_records(root: Path) -> List[CaseRecord]:
    """Load the paired IMA++ cohort using its prespecified binary labels."""
    frame = pd.read_csv(
        root / "metadata" / "clinical_binary.csv",
        dtype={"isic_id": str, "group_id": str},
    )
    required = {"isic_id", "group_id", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"IMAplusplus metadata missing required columns: {sorted(missing)}")
    if frame["isic_id"].duplicated().any():
        raise ValueError("IMAplusplus metadata contains duplicate isic_id values")
    records = []
    for row in frame.itertuples(index=False):
        pid = str(row.isic_id).strip()
        image = root / "images" / f"{pid}.jpg"
        mask = root / "selected_masks" / f"{pid}.png"
        if not image.is_file() or not mask.is_file():
            raise FileNotFoundError(
                f"IMAplusplus pair missing for {pid}: image={image.is_file()} mask={mask.is_file()}"
            )
        records.append(CaseRecord(pid, image, (mask,), int(row.label), str(row.group_id).strip()))
    return records


def load_records(
    dataset: str, data_root: Path, clinical_name: Union[str, Path] = "clinical.xlsx"
) -> List[CaseRecord]:
    dataset_key = dataset.lower()
    root_names = {"her2usc": "HER2", "lmnusc": "LNM", "breast": "BrEaST", "busi": "BUSI", "isic2018": "ISIC2018", "imaplusplus": "IMAplusplus"}
    if dataset_key not in root_names:
        raise ValueError(f"unknown dataset {dataset!r}")
    root = Path(data_root) / root_names[dataset_key]
    if dataset_key in {"her2usc", "lmnusc"}:
        records = _flat_records(root, clinical_name=clinical_name)
    elif dataset_key == "breast":
        records = _breast_records(root, clinical_name=clinical_name)
    elif dataset_key == "busi":
        records = _busi_records(root)
    elif dataset_key == "imaplusplus":
        records = _imaplusplus_records(root)
    else:
        records = _isic_records(root)
    if not records:
        raise RuntimeError(f"no matched cases found for {dataset} at {root}")
    return records


class ClinicalPreprocessor:
    def __init__(self, fixed_categories: Optional[Dict[str, Sequence[str]]] = None):
        self.numeric_columns, self.categorical_columns = [], []
        self.fixed_categories = {
            str(column): list(map(str, values))
            for column, values in (fixed_categories or {}).items()
        }

    def fit(self, frame: pd.DataFrame):
        self.numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = [c for c in frame.columns if c not in self.numeric_columns]
        self.means = frame[self.numeric_columns].mean().fillna(0.0)
        filled = frame[self.numeric_columns].fillna(self.means)
        self.stds = filled.std(ddof=0).replace(0, 1).fillna(1)
        self.categories = {}
        for column in self.categorical_columns:
            observed = frame[column].fillna("<MISSING>").astype(str).unique().tolist()
            values = self.fixed_categories.get(str(column), observed)
            self.categories[column] = sorted(set(values) | {"<MISSING>", "<UNK>"})
        self.feature_slices_, self.feature_names_ = [], []
        offset = 0
        for column in self.numeric_columns:
            self.feature_slices_.append((offset, offset + 1))
            self.feature_names_.append(str(column))
            offset += 1
        for column in self.categorical_columns:
            width = len(self.categories[column])
            self.feature_slices_.append((offset, offset + width))
            self.feature_names_.append(str(column))
            offset += width
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        numeric = ((frame[self.numeric_columns].fillna(self.means) - self.means) / self.stds).to_numpy(np.float32)
        parts = [numeric]
        for col in self.categorical_columns:
            values = frame[col].fillna("<MISSING>").astype(str)
            values = values.where(values.isin(self.categories[col]), "<UNK>")
            parts.append(np.stack([(values == category).to_numpy(np.float32) for category in self.categories[col]], axis=1))
        return np.concatenate(parts, axis=1) if parts else np.zeros((len(frame), 0), np.float32)

    def state_dict(self) -> dict:
        return {
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
            "means": self.means.to_dict(),
            "stds": self.stds.to_dict(),
            "categories": {str(key): list(value) for key, value in self.categories.items()},
            "feature_slices": [list(value) for value in self.feature_slices_],
            "feature_names": list(self.feature_names_),
        }


def load_clinical_frame(
    dataset: str,
    data_root: Path,
    records: Sequence[CaseRecord],
    clinical_name: Union[str, Path] = "clinical.xlsx",
) -> pd.DataFrame:
    key = dataset.lower()
    if key == "busi" or key == "isic2018":
        return pd.DataFrame(index=[r.pid for r in records])
    if key == "imaplusplus":
        frame = pd.read_csv(
            Path(data_root) / "IMAplusplus" / "metadata" / "clinical_binary.csv",
            dtype={"isic_id": str, "group_id": str},
        )
        frame["isic_id"] = frame["isic_id"].astype(str).str.strip()
        frame = frame.set_index("isic_id").drop(columns=["group_id", "label"])
        return frame.reindex([r.pid for r in records])
    root = Path(data_root) / ({"her2usc": "HER2", "lmnusc": "LNM", "breast": "BrEaST"}[key])
    frame = pd.read_excel(root / clinical_name, engine="openpyxl")
    pid_col = frame.columns[0]
    if key == "breast":
        excluded = {
            "image_filename", "mask_tumor_filename", "mask_other_filename",
            "classification", "verification", "diagnosis",
        }
        drop = [c for c in frame.columns if str(c).strip().casefold() in excluded]
    else:
        drop = [frame.columns[-1]] + (["ID", "\u6587\u5b57"] if key == "lmnusc" else [])
    frame[pid_col] = frame[pid_col].astype(str).str.strip()
    frame = frame.set_index(pid_col).drop(columns=[c for c in drop if c in frame.columns])
    return frame.reindex([r.pid for r in records])


class BaselineDataset(Dataset):
    def __init__(
        self,
        records: Sequence[CaseRecord],
        clinical: Optional[np.ndarray],
        image_size=256,
        augment=False,
        grayscale=True,
        transform=None,
    ):
        self.records, self.clinical, self.image_size = list(records), clinical, int(image_size)
        self.augment, self.grayscale = bool(augment), bool(grayscale)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        read_mode = cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR
        image = cv2.imread(str(record.image_path), read_mode)
        if image is None:
            raise FileNotFoundError(record.image_path)
        if self.grayscale:
            image = np.repeat(image[..., None], 3, axis=2)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        has_mask = bool(record.mask_paths)
        for path in record.mask_paths:
            current = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if current is not None:
                if current.shape != mask.shape:
                    current = cv2.resize(
                        current, (mask.shape[1], mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                mask = np.maximum(mask, (current > 0).astype(np.uint8))
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            if torch.rand(()) < 0.5:
                image, mask = np.fliplr(image).copy(), np.fliplr(mask).copy()
            if torch.rand(()) < 0.5:
                image, mask = np.flipud(image).copy(), np.flipud(mask).copy()
        image_t = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0
        mask_t = torch.from_numpy(mask.copy()).float().unsqueeze(0)
        if self.transform is not None:
            image_t, mask_t = self.transform(image_t, mask_t)
        clinical_t = torch.from_numpy(self.clinical[index]).float() if self.clinical is not None else torch.zeros(0)
        return {
            "pid": record.pid, "image": image_t, "mask": mask_t,
            "has_mask": torch.tensor(has_mask), "clinical": clinical_t, "label": torch.tensor(record.label, dtype=torch.long),
        }
