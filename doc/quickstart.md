# Quick start

BUA-LEL provides joint lesion segmentation and multimodal classification, comparative baselines, ablations, calibration, paired statistical analysis, and evidence visualization.

1. Install a matching PyTorch/torchvision pair, then run `pip install -r requirements.txt` and `pip install -e . --no-deps`.
2. Prepare local images, masks, and clinical tables using the [data guide](data.md), and obtain the MedSAM pretrained weights separately.
3. Set dataset and checkpoint paths in `configs/`.
4. Inspect the configuration with `python scripts/cross_validate.py --config configs/her2usc.yaml --dry-run`, then omit `--dry-run` to train.
5. Use `scripts/inference.py` for single-case prediction without reference labels and `scripts/evaluate.py` for evaluation on specified held-out cases.

See the [model description](model.md), [experimental protocols](experiments.md), and [baseline guide](baselines.md) for details.

## Datasets

HER2USC and LMNUSC are private clinical cohorts. BrEaST supports joint segmentation and binary diagnosis; BUSI supports image-only segmentation. IMA++ supports segmentation with binary or three-class diagnosis. ISIC2018 supports segmentation.

Prepare datasets and model weights locally using the [data guide](data.md) and the paths in your configuration.

BUA-LEL code uses the MIT license. Segment Anything retains Apache-2.0. Cite the software using `CITATION.cff` and cite the accompanying manuscript.
