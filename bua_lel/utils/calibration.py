from typing import Dict

import numpy as np


class BinaryCalibrationAccumulator:
    """Streaming fixed-bin calibration statistics for binary probabilities."""

    def __init__(self, n_bins: int = 15):
        if n_bins < 1:
            raise ValueError("n_bins must be positive")
        self.n_bins = int(n_bins)
        self.counts = np.zeros(self.n_bins, dtype=np.int64)
        self.confidence_sums = np.zeros(self.n_bins, dtype=np.float64)
        self.target_sums = np.zeros(self.n_bins, dtype=np.float64)
        self.squared_error_sum = 0.0
        self.total = 0

    def update(self, probabilities, targets) -> None:
        probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        targets = np.asarray(targets, dtype=np.float64).reshape(-1)
        if probabilities.shape != targets.shape:
            raise ValueError("probabilities and targets must have the same shape")
        if probabilities.size == 0:
            return
        if not np.isfinite(probabilities).all() or not np.isfinite(targets).all():
            raise ValueError("calibration inputs must be finite")
        probabilities = np.clip(probabilities, 0.0, 1.0)
        bin_ids = np.minimum((probabilities * self.n_bins).astype(np.int64), self.n_bins - 1)
        self.counts += np.bincount(bin_ids, minlength=self.n_bins)
        self.confidence_sums += np.bincount(bin_ids, weights=probabilities, minlength=self.n_bins)
        self.target_sums += np.bincount(bin_ids, weights=targets, minlength=self.n_bins)
        self.squared_error_sum += float(np.square(probabilities - targets).sum())
        self.total += int(probabilities.size)

    def compute(self) -> Dict[str, object]:
        confidence = np.divide(
            self.confidence_sums,
            self.counts,
            out=np.full(self.n_bins, np.nan),
            where=self.counts > 0,
        )
        accuracy = np.divide(
            self.target_sums,
            self.counts,
            out=np.full(self.n_bins, np.nan),
            where=self.counts > 0,
        )
        if self.total == 0:
            ece = brier = float("nan")
        else:
            occupied = self.counts > 0
            weights = self.counts[occupied] / self.total
            ece = float(np.sum(weights * np.abs(accuracy[occupied] - confidence[occupied])))
            brier = float(self.squared_error_sum / self.total)
        return {
            "ece": ece,
            "brier_score": brier,
            "n_pixels": self.total,
            "bin_count": self.counts.copy(),
            "bin_confidence": confidence,
            "bin_accuracy": accuracy,
            "bin_lower": np.arange(self.n_bins) / self.n_bins,
            "bin_upper": np.arange(1, self.n_bins + 1) / self.n_bins,
        }
