import os
import warnings
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


class DualTaskDataset(Dataset):
    """DualTaskDataset."""

    IMG_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")

    def __init__(
        self,
        image_dir: str,
        mask_dir: Optional[str],
        clinical_excel: str,
        mode: str = "seg",
        transform=None,
        image_size: Tuple[int, int] = (256, 256),
        allow_missing_mask: bool = False,
        return_pid: bool = False,


        num_scaler: Optional[Dict[str, np.ndarray]] = None,
        cat_maps: Optional[Dict[str, Dict[str, int]]] = None,
        unknown_cat_as_missing: bool = False,
        return_cat_targets: bool = True,
        exclude_feature_columns: Optional[List[str]] = None,
        clinical_preprocessing_profile: str = "missing_aware",
        image_mode: str = "grayscale",
    ):
        self.mode = str(mode)
        if self.mode not in ("seg", "cls"):
            raise ValueError(f"mode must be 'seg' or 'cls', got {mode}")

        self.transform = transform

        self.H, self.W = int(image_size[0]), int(image_size[1])
        self.allow_missing_mask = bool(allow_missing_mask)
        self.return_pid = bool(return_pid)

        self.num_scaler = num_scaler
        self.cat_maps = cat_maps
        self.unknown_cat_as_missing = bool(unknown_cat_as_missing)
        self.exclude_feature_columns = set(exclude_feature_columns or [])
        self.return_cat_targets = bool(return_cat_targets)
        self.clinical_preprocessing_profile = str(clinical_preprocessing_profile)
        self.image_mode = str(image_mode).lower()
        if self.image_mode not in {"grayscale", "rgb"}:
            raise ValueError("image_mode must be 'grayscale' or 'rgb'")
        if self.clinical_preprocessing_profile not in {
            "missing_aware", "complete_observed",
        }:
            raise ValueError(
                "clinical_preprocessing_profile must be 'missing_aware' or "
                f"'complete_observed', got {clinical_preprocessing_profile!r}"
            )

        if not clinical_excel or (not os.path.exists(clinical_excel)):
            raise ValueError("A valid clinical_excel path must be provided.")
        self.clinical_excel = clinical_excel

        if not image_dir or (not os.path.exists(image_dir)):
            raise ValueError("A valid image_dir path must be provided.")
        self.image_dir = image_dir

        self.mask_dir = mask_dir
        if self.mode == "seg":
            if (not self.mask_dir) or (not os.path.exists(self.mask_dir)):
                raise ValueError("seg mode requires a valid mask_dir.")

        (
            self.patient_ids,
            self.c_obs_all,
            self.m_all,
            self.all_labels,
            self.feature_names,
            self.numeric_feature_names,
            self.onehot_feature_names,
            self.onehot_slices,
            self.numeric_slice,
            self.cat_targets_all,
        ) = self._extract_excel_features(
            self.clinical_excel,
            num_scaler=self.num_scaler,
            cat_maps=self.cat_maps,
            return_cat_targets=self.return_cat_targets,
        )

        self.pid_to_idx = {pid: idx for idx, pid in enumerate(self.patient_ids)}

        self.image_path_dict = self._scan_paths(self.image_dir, exts=self.IMG_EXTS)

        if self.mask_dir and os.path.exists(self.mask_dir):
            self.mask_path_dict = self._scan_paths(
                self.mask_dir,
                exts=(".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"),
            )
        else:
            self.mask_path_dict = {}


        self.valid_pids = [pid for pid in self.image_path_dict.keys() if pid in self.pid_to_idx]


        if self.mode == "seg" and (not self.allow_missing_mask):
            self.valid_pids = [pid for pid in self.valid_pids if pid in self.mask_path_dict]

        if len(self.valid_pids) == 0:
            raise RuntimeError(
                "No valid samples found. Please check whether image/mask filenames "
                "match the patient IDs in clinical Excel."
            )

        self._print_stats()


    @staticmethod
    def norm_pid(x: Any) -> str:
        return str(x).strip()


    def get_feature_dim(self) -> int:
        return int(self.c_obs_all.shape[1])

    def get_pid_list(self) -> List[str]:
        return list(self.valid_pids)


    def _scan_paths(self, root_dir: str, exts: Tuple[str, ...]) -> Dict[str, str]:
        d: Dict[str, str] = {}
        for fn in os.listdir(root_dir):
            if fn.lower().endswith(exts):
                pid = self.norm_pid(os.path.splitext(fn)[0])
                d[pid] = os.path.join(root_dir, fn)
        if len(d) == 0:
            raise ValueError(f"No supported files found in directory: {root_dir}")
        return d


    def _print_stats(self):
        binc = np.bincount(self.all_labels, minlength=3)

        print(f"\n📊 {self.mode.upper()} mode - BUA-LEL Dataset without clinical missingness:")
        print(f"   - Clinical rows: {len(self.patient_ids)}")
        print(f"   - Image files: {len(self.image_path_dict)}")
        print(f"   - Mask files: {len(self.mask_path_dict)}")
        print(f"   - Matched valid samples: {len(self.valid_pids)}")
        print(f"   - Clinical feature dim: {self.get_feature_dim()}")
        print(f"   - Label distribution: {binc} for labels 0/1/2")

        if self.num_scaler is None:
            print("   - Numeric standardization: not applied (num_scaler=None)")
        else:
            print("   - Numeric standardization: applied with external fold-wise num_scaler")

        if self.cat_maps is None:
            print("   - Category maps: not provided; built from the current Excel file")
            print("     Warning: this is only recommended for full-data/inspection runs.")
        else:
            print("   - Category maps: provided externally for fold-wise no-leak transform")

        print("   - Missing values: fold-train mean/<MISSING> imputation; m is all-one after imputation")


    def _convert_labels(self, raw_labels: np.ndarray) -> np.ndarray:
        s = pd.Series(raw_labels)

        if s.isna().any():
            bad_idx = s[s.isna()].index.tolist()[:10]
            raise ValueError(f"Label column contains NaN. Example row indices: {bad_idx}")

        try:
            labels = s.astype(float).astype(int).to_numpy()
        except Exception:
            s_str = s.astype(str).str.strip().str.upper()
            mapping = {
                "LN0": 0,
                "LN1-3": 1,
                "LN1–3": 1,
                "LN1—3": 1,
                "LN4+": 2,
                "LN4 +": 2,
                "HER2-ZERO": 0,
                "HER2_ZERO": 0,
                "HER2ZERO": 0,
                "ZERO": 0,
                "HER2-LOW": 1,
                "HER2_LOW": 1,
                "HER2LOW": 1,
                "LOW": 1,
                "HER2-POSITIVE": 2,
                "HER2_POSITIVE": 2,
                "HER2POSITIVE": 2,
                "POSITIVE": 2,
            }
            mapped = s_str.map(mapping)
            if mapped.isna().any():
                bad = s_str[mapped.isna()].value_counts().head(20)
                raise ValueError(
                    "Label column contains unrecognized string labels.\n"
                    f"Unrecognized labels (Top20):\n{bad}"
                )
            labels = mapped.astype(int).to_numpy()

        u = np.unique(labels)
        if np.any(u < 0):
            raise ValueError(f"Invalid labels found: {u}. Labels must be non-negative integers.")
        return labels.astype(np.int64)


    @staticmethod
    def _apply_num_scaler(
        num_obs: np.ndarray,
        num_scaler: Optional[Dict[str, np.ndarray]],
        complete_observed: bool = False,
    ) -> np.ndarray:
        """
        Apply externally fitted fold-wise mean imputation and z-score standardization.
        """
        if num_scaler is None:
            return num_obs.astype(np.float32)

        mean = np.asarray(num_scaler["mean"], dtype=np.float32)
        std = np.asarray(num_scaler["std"], dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

        if num_obs.shape[1] != mean.shape[0]:
            raise ValueError(
                f"num_scaler dimension mismatch: numeric dim={num_obs.shape[1]}, "
                f"mean dim={mean.shape[0]}"
            )

        if complete_observed:
            if np.isnan(num_obs).any():
                raise ValueError("complete_observed does not permit missing numeric clinical values")
            z = (num_obs - mean[None, :]) / std[None, :]
        else:
            filled = np.where(np.isnan(num_obs), mean[None, :], num_obs)
            z = (filled - mean[None, :]) / std[None, :]
        return z.astype(np.float32)


    def _extract_excel_features(
        self,
        filename: str,
        num_scaler: Optional[Dict[str, np.ndarray]],
        cat_maps: Optional[Dict[str, Dict[str, int]]],
        return_cat_targets: bool,
        raw_frame: Optional[pd.DataFrame] = None,
    ):
        suffix = os.path.splitext(str(filename))[1].lower()


        if raw_frame is not None:
            df = raw_frame.copy(deep=True)
        elif suffix == ".csv":
            df = pd.read_csv(filename)
        elif suffix == ".tsv":
            df = pd.read_csv(filename, sep="\t")
        else:
            df = pd.read_excel(filename, engine="openpyxl")

        print(f"\n🔍 Processing clinical data: {filename}")
        print(f"   - Excel rows: {len(df)}")
        print(f"   - Excel columns: {len(df.columns)}")

        if len(df.columns) < 2:
            raise ValueError(
                "clinical_excel must contain at least 2 columns: "
                "patient_id, optional clinical features, and label."
            )

        patient_ids = df.iloc[:, 0].astype(str).apply(self.norm_pid).to_numpy()
        labels = self._convert_labels(df.iloc[:, -1].to_numpy())

        features_df = df.iloc[:, 1:-1].copy()
        features_df = features_df.drop(
            columns=[c for c in self.exclude_feature_columns if c in features_df.columns]
        )
        print(f"   - Raw feature columns: {len(features_df.columns)}")
        complete_observed = self.clinical_preprocessing_profile == "complete_observed"
        if complete_observed and features_df.isna().any().any():
            na_counts = features_df.isna().sum()
            bad = na_counts[na_counts > 0].sort_values(ascending=False)
            raise ValueError(
                "complete_observed requires a clinical table without missing values. "
                f"Columns with missing values:\n{bad}"
            )

        numeric_df = features_df.select_dtypes(include=[np.number]).copy()
        cat_df = features_df.select_dtypes(exclude=[np.number]).copy()

        num_cols = list(numeric_df.columns)
        cat_cols = list(cat_df.columns)

        print(f"   - Numeric feature columns: {len(num_cols)}")
        print(f"   - Categorical feature columns: {len(cat_cols)}")


        if len(num_cols) > 0:
            num_obs = numeric_df.to_numpy(dtype=np.float32)
            if np.isnan(num_obs).any() and num_scaler is None:
                raise ValueError("num_scaler fitted on the training fold is required when numeric values are missing")
            num_obs = self._apply_num_scaler(
                num_obs,
                num_scaler,
                complete_observed=complete_observed,
            )
        else:
            num_obs = np.zeros((len(df), 0), dtype=np.float32)

        numeric_slice = (0, len(num_cols))
        numeric_feature_names = [str(c) for c in num_cols]


        if cat_maps is None:
            cat_maps = {}
            for col in cat_cols:
                if complete_observed:
                    cats = pd.Series(cat_df[col].astype(str).unique()).sort_values().tolist()
                else:
                    cats = pd.Series(cat_df[col].fillna("<MISSING>").astype(str).unique()).sort_values().tolist()
                    if "<MISSING>" not in cats:
                        cats.append("<MISSING>")
                    if "<UNK>" not in cats:
                        cats.append("<UNK>")
                cat_maps[col] = {c: i for i, c in enumerate(cats)}

        onehot_obs_list: List[np.ndarray] = []
        onehot_feature_names: List[str] = []
        onehot_slices: Dict[str, Tuple[int, int]] = {}
        cat_targets_all: Dict[str, np.ndarray] = {}


        cursor = len(num_cols)

        for col in cat_cols:
            mapper = cat_maps.get(col, {})
            K = int(len(mapper))

            if K <= 0:
                raise ValueError(f"Empty category map for column: {col}")

            s = (
                cat_df[col].astype(str)
                if complete_observed
                else cat_df[col].fillna("<MISSING>").astype(str)
            )
            oh = np.zeros((len(df), K), dtype=np.float32)
            targets = np.zeros((len(df),), dtype=np.int64) if return_cat_targets else None

            for r, v in enumerate(s.to_numpy()):
                j = mapper.get(v, None if complete_observed else mapper.get("<UNK>"))
                if j is None:
                    raise ValueError(
                        f"Unknown category found during transform: col={col}, value={v}. "
                        "Please fit cat_maps on the training fold and ensure val/test "
                        "categories are covered, or clean the category values."
                    )

                oh[r, j] = 1.0
                if targets is not None:
                    targets[r] = int(j)

            onehot_slices[col] = (cursor, cursor + K)
            cursor += K

            for c, j in sorted(mapper.items(), key=lambda x: x[1]):
                onehot_feature_names.append(f"{col}__{c}")

            onehot_obs_list.append(oh)

            if targets is not None:
                cat_targets_all[col] = targets

        onehot_obs = (
            np.concatenate(onehot_obs_list, axis=1).astype(np.float32)
            if len(onehot_obs_list) > 0
            else np.zeros((len(df), 0), dtype=np.float32)
        )


        c_obs = np.concatenate([num_obs, onehot_obs], axis=1).astype(np.float32)


        m = np.ones_like(c_obs, dtype=np.float32)

        feature_names = numeric_feature_names + onehot_feature_names

        print(f"   - Final clinical feature dim: {c_obs.shape[1]}")
        if complete_observed:
            print("   - Clinical preprocessing: complete-observed fold transformation")
        else:
            print("   - Missing clinical values: fold-train mean/<MISSING> imputation; m entries are 1 after imputation")

        return (
            patient_ids,
            c_obs,
            m,
            labels,
            feature_names,
            numeric_feature_names,
            onehot_feature_names,
            onehot_slices,
            numeric_slice,
            cat_targets_all,
        )


    def _read_image(self, img_path: str) -> torch.Tensor:
        read_flag = cv2.IMREAD_COLOR if self.image_mode == "rgb" else cv2.IMREAD_GRAYSCALE
        img = cv2.imread(img_path, read_flag)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if self.image_mode == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        if self.image_mode == "rgb":
            img_t = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        else:

            img_t = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1)
        return img_t

    def _read_mask(self, mask_path: str) -> torch.Tensor:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")

        mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)

        return torch.from_numpy(mask).unsqueeze(0)

    def __len__(self) -> int:
        return len(self.valid_pids)

    def __getitem__(self, idx: int):
        pid = self.valid_pids[idx]
        cidx = self.pid_to_idx[pid]

        img = self._read_image(self.image_path_dict[pid])

        if self.mode == "seg":
            mask_path = self.mask_path_dict.get(pid, None)
            if mask_path is not None:
                seg_mask = self._read_mask(mask_path)
                has_mask = torch.tensor(1, dtype=torch.uint8)
            else:
                if not self.allow_missing_mask:
                    raise FileNotFoundError(f"seg mode missing mask: pid={pid}")
                seg_mask = torch.zeros((1, self.H, self.W), dtype=torch.float32)
                has_mask = torch.tensor(0, dtype=torch.uint8)
        else:
            seg_mask = torch.zeros((1, self.H, self.W), dtype=torch.float32)
            has_mask = torch.tensor(0, dtype=torch.uint8)

        c_obs = torch.tensor(self.c_obs_all[cidx], dtype=torch.float32)


        m = torch.tensor(self.m_all[cidx], dtype=torch.float32)

        y = torch.tensor(self.all_labels[cidx], dtype=torch.long)

        cat_targets = None
        if self.return_cat_targets:
            cat_targets = {
                col: torch.tensor(int(arr[cidx]), dtype=torch.long)
                for col, arr in self.cat_targets_all.items()
            }

        if self.transform is not None:

            img, seg_mask = self.transform(img, seg_mask)

        if self.return_pid:
            if self.return_cat_targets:
                return img, seg_mask, has_mask, c_obs, m, y, cat_targets, pid
            return img, seg_mask, has_mask, c_obs, m, y, pid

        if self.return_cat_targets:
            return img, seg_mask, has_mask, c_obs, m, y, cat_targets

        return img, seg_mask, has_mask, c_obs, m, y
