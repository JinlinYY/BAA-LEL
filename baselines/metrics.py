from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def classification_metrics(
    y_true: Sequence[int], y_proba: np.ndarray, class_names: Optional[Sequence[str]] = None
) -> Dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    if y_proba.ndim != 2 or len(y_true) != len(y_proba):
        raise ValueError("y_proba must be [N,C] and aligned with y_true")
    if not np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("y_proba rows must sum to one")
    n_classes = y_proba.shape[1]
    names = list(class_names or [str(i) for i in range(n_classes)])
    if len(names) != n_classes:
        raise ValueError("class_names length must equal y_proba columns")

    y_pred = y_proba.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    per_class = {}
    aucs, sensitivities, specificities = [], [], []
    for c, name in enumerate(names):
        tp = float(cm[c, c])
        fn = float(cm[c, :].sum() - tp)
        fp = float(cm[:, c].sum() - tp)
        tn = float(cm.sum() - tp - fn - fp)
        sensitivity = tp / (tp + fn) if tp + fn else float("nan")
        specificity = tn / (tn + fp) if tn + fp else float("nan")
        binary_true = (y_true == c).astype(np.uint8)
        try:
            auc = float(roc_auc_score(binary_true, y_proba[:, c]))
        except ValueError:
            auc = float("nan")
        per_class[str(name)] = {"AUC": auc, "Sensitivity": sensitivity, "Specificity": specificity}
        aucs.append(auc)
        sensitivities.append(sensitivity)
        specificities.append(specificity)

    return {
        "ACC": float(accuracy_score(y_true, y_pred)),
        "Macro-F1": float(f1_score(y_true, y_pred, labels=np.arange(n_classes), average="macro", zero_division=0)),
        "Macro-AUC": float(np.nanmean(aucs)),
        "Macro-Sensitivity": float(np.nanmean(sensitivities)),
        "Macro-Specificity": float(np.nanmean(specificities)),
        "per_class": per_class,
    }


def _surface(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3,) * mask.ndim, dtype=bool)
    return np.logical_xor(mask, binary_erosion(mask, structure=structure, border_value=0))


def segmentation_case_metrics(
    prediction: np.ndarray, target: np.ndarray, spacing: Optional[Tuple[float, ...]] = None
) -> Dict[str, float]:
    pred = np.asarray(prediction, dtype=bool).squeeze()
    true = np.asarray(target, dtype=bool).squeeze()
    if pred.shape != true.shape or pred.ndim not in (2, 3):
        raise ValueError("prediction and target must be aligned 2D or 3D masks")
    spacing = tuple(spacing or (1.0,) * pred.ndim)
    if len(spacing) != pred.ndim:
        raise ValueError("spacing dimensionality must match masks")

    intersection = np.logical_and(pred, true).sum(dtype=np.float64)
    pred_sum, true_sum = pred.sum(dtype=np.float64), true.sum(dtype=np.float64)
    union = np.logical_or(pred, true).sum(dtype=np.float64)
    dice = 1.0 if pred_sum + true_sum == 0 else 2.0 * intersection / (pred_sum + true_sum)
    iou = 1.0 if union == 0 else intersection / union

    if pred_sum == 0 and true_sum == 0:
        hd95 = assd = 0.0
    elif pred_sum == 0 or true_sum == 0:
        hd95 = assd = float(np.linalg.norm(np.asarray(pred.shape) * np.asarray(spacing)))
    else:
        pred_surface, true_surface = _surface(pred), _surface(true)
        distance_to_true = distance_transform_edt(~true_surface, sampling=spacing)[pred_surface]
        distance_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)[true_surface]
        distances = np.concatenate([distance_to_true, distance_to_pred])
        hd95 = float(np.percentile(distances, 95))
        assd = float((distance_to_true.mean() + distance_to_pred.mean()) / 2.0)
    return {"Dice": float(dice), "mIoU": float(iou), "HD95": hd95, "ASSD": assd}
