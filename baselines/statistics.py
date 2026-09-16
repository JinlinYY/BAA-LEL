from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from baselines.metrics import classification_metrics


def paired_bootstrap_compare(
    method: pd.DataFrame, baseline: pd.DataFrame, value_col: str,
    n_bootstrap: int = 10000, seed: int = 42,
) -> Dict[str, float]:
    paired = method[["pid", value_col]].merge(
        baseline[["pid", value_col]], on="pid", how="inner", suffixes=("_method", "_baseline"), validate="one_to_one"
    ).dropna()
    if len(paired) < 2:
        raise ValueError("at least two paired patients are required")
    method_values = paired[f"{value_col}_method"].to_numpy()
    baseline_values = paired[f"{value_col}_baseline"].to_numpy()
    differences = method_values - baseline_values
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(differences), size=(int(n_bootstrap), len(differences)))
    method_samples = method_values[sample_indices].mean(axis=1)
    baseline_samples = baseline_values[sample_indices].mean(axis=1)
    samples = method_samples - baseline_samples
    try:
        p_value = float(wilcoxon(differences, zero_method="zsplit", alternative="two-sided").pvalue)
    except ValueError:
        p_value = 1.0
    return {
        "n_pairs": int(len(differences)), "method": float(method_values.mean()), "baseline": float(baseline_values.mean()),
        "difference": float(differences.mean()),
        "method_ci_low": float(np.percentile(method_samples, 2.5)), "method_ci_high": float(np.percentile(method_samples, 97.5)),
        "baseline_ci_low": float(np.percentile(baseline_samples, 2.5)), "baseline_ci_high": float(np.percentile(baseline_samples, 97.5)),
        "difference_ci_low": float(np.percentile(samples, 2.5)), "difference_ci_high": float(np.percentile(samples, 97.5)),
        "p_value_wilcoxon": p_value,
    }


def paired_classification_compare(
    method: pd.DataFrame, baseline: pd.DataFrame, class_names: Sequence[str], metric: str = "Macro-AUC",
    n_bootstrap: int = 10000, n_permutations: int = 10000, seed: int = 42,
) -> Dict[str, float]:
    probability_columns = [f"proba_{name}" for name in class_names]
    columns = ["pid", "y_true", *probability_columns]
    paired = method[columns].merge(
        baseline[columns], on="pid", suffixes=("_method", "_baseline"), validate="one_to_one"
    )
    y_method = paired["y_true_method"].to_numpy(dtype=int)
    y_baseline = paired["y_true_baseline"].to_numpy(dtype=int)
    if not np.array_equal(y_method, y_baseline):
        raise ValueError("paired OOF files disagree on y_true")
    method_proba = paired[[f"{c}_method" for c in probability_columns]].to_numpy()
    baseline_proba = paired[[f"{c}_baseline" for c in probability_columns]].to_numpy()

    def score(indices, proba):
        return float(classification_metrics(y_method[indices], proba[indices], class_names)[metric])

    full = np.arange(len(paired))
    method_score, baseline_score = score(full, method_proba), score(full, baseline_proba)
    observed = method_score - baseline_score
    rng = np.random.default_rng(seed)
    method_boot, baseline_boot, difference_boot = [], [], []
    class_indices = [np.flatnonzero(y_method == class_id) for class_id in np.unique(y_method)]
    for _ in range(int(n_bootstrap)):

        indices = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in class_indices])
        try:
            a, b = score(indices, method_proba), score(indices, baseline_proba)
        except ValueError:
            continue
        if np.isfinite(a) and np.isfinite(b):
            method_boot.append(a); baseline_boot.append(b); difference_boot.append(a - b)
    if not difference_boot:
        raise ValueError("bootstrap samples could not produce a finite metric")
    extreme = 0
    for _ in range(int(n_permutations)):
        swap = rng.random(len(paired)) < 0.5
        perm_a, perm_b = method_proba.copy(), baseline_proba.copy()
        perm_a[swap], perm_b[swap] = baseline_proba[swap], method_proba[swap]
        if abs(score(full, perm_a) - score(full, perm_b)) >= abs(observed) - 1e-12:
            extreme += 1
    return {
        "n_pairs": int(len(paired)), "method": method_score, "baseline": baseline_score, "difference": observed,
        "method_ci_low": float(np.percentile(method_boot, 2.5)), "method_ci_high": float(np.percentile(method_boot, 97.5)),
        "baseline_ci_low": float(np.percentile(baseline_boot, 2.5)), "baseline_ci_high": float(np.percentile(baseline_boot, 97.5)),
        "difference_ci_low": float(np.percentile(difference_boot, 2.5)), "difference_ci_high": float(np.percentile(difference_boot, 97.5)),
        "p_value_paired_permutation": float((extreme + 1) / (int(n_permutations) + 1)),
    }
