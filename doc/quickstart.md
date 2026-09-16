# Quick start

BUA-LEL provides joint lesion segmentation and multimodal classification, comparative baselines, ablations, calibration, paired statistical analysis, and evidence visualization.

1. Install a matching PyTorch/torchvision pair, then run `pip install -r requirements.txt` and `pip install -e . --no-deps`.
2. Prepare local images, masks, and clinical tables using the [data guide](data.md), and obtain the MedSAM pretrained weights separately.
3. Set dataset and checkpoint paths in `configs/`.
4. Inspect the configuration with `python scripts/cross_validate.py --config configs/her2usc.yaml --dry-run`, then omit `--dry-run` to train.
5. Use `scripts/inference.py` for single-case prediction without reference labels and `scripts/evaluate.py` for evaluation on specified held-out cases.

See the [model description](model.md), [experimental protocols](experiments.md), and [baseline guide](baselines.md) for details.

## Data and reproducibility

HER2USC and LMNUSC are private clinical cohorts. BrEaST supports joint segmentation and binary diagnosis; BUSI supports image-only segmentation. The authors identified IMA++ as the actual source of the cross-domain experiment labeled ISIC2018 in the supplied manuscript. Use the IMA++ configurations for that dataset; the ISIC2018 segmentation configuration is separate.

Datasets, clinical records, patient splits, trained weights, and the manuscript PDF are not distributed. The runnable configurations do not certify numerical reproduction of all manuscript tables. Fold-validation checkpoint selection and nested evaluation are distinct protocols. The IMA++ configurations use ResNet-50, while the manuscript's common settings describe MedSAM. These distinctions are documented in the experimental guide.

The clinical encoder initializes each of its 12 prior graph nodes with the complete clinical vector; it does not assign one measured scalar to each node. Consider this implementation detail when interpreting node representations or adapting the model to other cohorts.

BUA-LEL code uses the MIT license. Segment Anything retains Apache-2.0. Cite the software using `CITATION.cff` and cite the accompanying manuscript.
