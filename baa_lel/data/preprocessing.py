import os
from typing import Dict, Optional, List, Tuple, Set

import numpy as np
import pandas as pd
from torch.utils.data import Subset

from baa_lel.data import DualTaskDataset

def norm_pid(x) -> str:
    return str(x).strip()

def scan_image_pids(image_dir: str) -> List[str]:
    exts = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
    pids = []
    for fn in os.listdir(image_dir):
        if fn.lower().endswith(exts):
            pids.append(norm_pid(os.path.splitext(fn)[0]))
    return pids

def read_excel_df(clinical_excel: str) -> pd.DataFrame:
    """Read excel df."""
    suffix = os.path.splitext(str(clinical_excel))[1].lower()
    if suffix == ".csv":
        df = pd.read_csv(clinical_excel)
    elif suffix == ".tsv":
        df = pd.read_csv(clinical_excel, sep="\t")
    else:
        df = pd.read_excel(clinical_excel, engine="openpyxl")
    df.iloc[:, 0] = df.iloc[:, 0].astype(str).apply(norm_pid)
    return df

def convert_labels(raw_labels: np.ndarray) -> np.ndarray:
    s = pd.Series(raw_labels)
    if s.isna().any():
        raise ValueError("The label column contains NaN values")
    try:
        labels = s.astype(float).astype(int).to_numpy()
    except Exception:
        s_str = s.astype(str).str.strip().str.upper()
        mapping = {"LN0": 0, "LN1-3": 1, "LN1–3": 1, "LN1—3": 1, "LN4+": 2, "LN4 +": 2}
        mapped = s_str.map(mapping)
        if mapped.isna().any():
            bad = s_str[mapped.isna()].value_counts().head(20)
            raise ValueError("Unrecognized labels: \n" + str(bad))
        labels = mapped.astype(int).to_numpy()

    if np.any(labels < 0):
        raise ValueError(f"Labels must be non-negative integers; found: {np.unique(labels)}")
    return labels

def build_pid_and_labels(image_dir: str, clinical_excel: str) -> Tuple[List[str], np.ndarray]:
    df = read_excel_df(clinical_excel)
    pid_col = df.columns[0]
    label_col = df.columns[-1]

    pid2y = {}
    y_arr = convert_labels(df[label_col].to_numpy())
    for pid, y in zip(df[pid_col].to_numpy(), y_arr):
        pid2y[str(pid).strip()] = int(y)

    img_pids = scan_image_pids(image_dir)
    valid_pids = [pid for pid in img_pids if pid in pid2y]
    y_all = np.array([pid2y[pid] for pid in valid_pids], dtype=np.int64)
    return valid_pids, y_all

def fit_fold_cat_maps(
    clinical_excel: str,
    train_pids: Set[str],
    exclude_columns=(),
    clinical_preprocessing_profile: str = "missing_aware",
) -> Dict[str, Dict[str, int]]:
    df = read_excel_df(clinical_excel)
    pid_col = df.columns[0]
    df_tr = df[df[pid_col].isin(train_pids)].copy()

    feats = df_tr.iloc[:, 1:-1].drop(columns=[c for c in exclude_columns if c in df_tr.columns])
    cat_df = feats.select_dtypes(exclude=[np.number])

    if clinical_preprocessing_profile not in {"missing_aware", "complete_observed"}:
        raise ValueError(f"Unknown clinical preprocessing profile: {clinical_preprocessing_profile}")
    complete_observed = clinical_preprocessing_profile == "complete_observed"
    if complete_observed and feats.isna().any().any():
        raise ValueError("complete_observed requires a clinical table without missing values")

    cat_maps: Dict[str, Dict[str, int]] = {}
    for col in cat_df.columns:
        if complete_observed:
            obs = cat_df[col].astype(str)
            cats = sorted(obs.unique().tolist())
        else:
            obs = cat_df[col].fillna("<MISSING>").astype(str)
            cats = sorted(set(obs.unique().tolist()) | {"<MISSING>", "<UNK>"})
        cat_maps[col] = {c: i for i, c in enumerate(cats)}
    return cat_maps

def fit_fold_num_scaler(clinical_excel: str, train_pids: Set[str], exclude_columns=()) -> Optional[Dict[str, np.ndarray]]:
    df = read_excel_df(clinical_excel)
    pid_col = df.columns[0]
    df_tr = df[df[pid_col].isin(train_pids)].copy()

    feats = df_tr.iloc[:, 1:-1].drop(columns=[c for c in exclude_columns if c in df_tr.columns])
    num_df = feats.select_dtypes(include=[np.number])

    if num_df.shape[1] == 0:
        return None

    means, stds = [], []
    for col in num_df.columns:
        vals = num_df[col].to_numpy(dtype=np.float32)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            mu, sd = 0.0, 1.0
        else:
            mu = float(vals.mean())
            sd = float(vals.std(ddof=0))
            if sd < 1e-6:
                sd = 1.0
        means.append(mu)
        stds.append(sd)
    return {"mean": np.array(means, dtype=np.float32), "std": np.array(stds, dtype=np.float32)}

def subset_by_pid_set(ds: DualTaskDataset, pid_set: Set[str]) -> Subset:
    idxs = [i for i, pid in enumerate(ds.valid_pids) if pid in pid_set]
    if len(idxs) == 0:
        raise RuntimeError("No matching cases in subset_by_pid_set; check the case identifier intersection")
    return Subset(ds, idxs)
