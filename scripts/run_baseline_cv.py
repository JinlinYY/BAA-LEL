import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.runner import BenchmarkConfig, run_benchmark


def main():
    parser = argparse.ArgumentParser(description="Leakage-safe five-fold BUA-LEL baseline benchmark")
    parser.add_argument("--dataset", required=True, choices=["HER2USC", "LMNUSC", "BrEaST", "BUSI", "ISIC2018", "IMAplusplus"])
    parser.add_argument("--method", required=True)
    parser.add_argument("--task", required=True, choices=["segmentation", "classification"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="outputs/baselines")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--inner-validation-fraction", type=float, default=.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--no-image-pretrained", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    values = vars(args)
    values["amp"] = not values.pop("no_amp")
    values["image_pretrained"] = not values.pop("no_image_pretrained")
    print(json.dumps(run_benchmark(BenchmarkConfig(**values)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
