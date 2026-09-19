# Third-party components and architectural references

## Segment Anything

`third_party/segment_anything/` contains an adapted Segment Anything package from Meta Platforms, Inc. and affiliates, under **Apache-2.0**. Its copyright headers, [license](third_party/segment_anything/LICENSE), and [notice](third_party/segment_anything/NOTICE) are retained. The local adaptation uses package-relative imports and explicit checkpoint loading. Source: https://github.com/facebookresearch/segment-anything.

## Pretrained encoders

MedSAM and torchvision pretrained models are external dependencies. No encoder weights are redistributed. Users obtain the parameters separately under their providers' terms. The BAA-LEL adapter and task modules are provided under this repository's MIT license.

## Baseline implementations

The baseline implementations include paper-based models and adaptations. Architecture references are listed in [doc/baselines.md](doc/baselines.md) and `baselines/models/__init__.py`. TransUNet, AAU-Net, HyperFusion, and MAP use independent adaptations. External repositories retain their respective licenses.

## Data and figures

Datasets, annotations, clinical records, source manuscript PDF, case-level results, and model weights are not distributed. `assets/framework.png` illustrates the BAA-LEL architecture. Dataset access and reuse remain governed by the respective providers.
