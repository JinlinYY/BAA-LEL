"""Run the formal BAA-LEL or MTANet nested five-fold protocol."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.multitask_runner import MultitaskConfig, run_multitask_benchmark


def main():
    parser = argparse.ArgumentParser(description="Formal joint segmentation-classification CV")
    parser.add_argument("--dataset", required=True, choices=["HER2USC", "LMNUSC", "BrEaST", "BUSI", "IMAplusplus"])
    parser.add_argument("--method", required=True, choices=["baa_lel", "medsam_mtl", "mtanet"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="outputs/baselines")
    parser.add_argument("--split-root")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=256, choices=[256])
    parser.add_argument("--medsam-input-size", type=int, default=1024, choices=[1024])
    parser.add_argument("--inner-validation-fraction", type=float, default=.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args())
    args["amp"] = not args.pop("no_amp")
    print(json.dumps(run_multitask_benchmark(MultitaskConfig(**args)), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
