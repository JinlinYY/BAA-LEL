# Third-party components and architectural references

## Segment Anything

`third_party/segment_anything/` contains an adapted Segment Anything package from Meta Platforms, Inc. and affiliates, under **Apache-2.0**. Its copyright headers, [license](third_party/segment_anything/LICENSE), and [notice](third_party/segment_anything/NOTICE) are retained. The local adaptation uses package-relative imports and explicit checkpoint loading. Source: https://github.com/facebookresearch/segment-anything.

## Pretrained encoders

MedSAM and torchvision pretrained models are external dependencies. No encoder weights are redistributed. Users obtain the parameters separately under their providers' terms. The BUA-LEL adapter and task modules are provided under this repository's MIT license.

## Baseline implementations

The baseline implementations comprise local paper-based models and source-informed adaptations. Their architectural references and assumptions are documented in [doc/baselines.md](doc/baselines.md) and `baselines/models/__init__.py`. The source-informed TransUNet adapter is independently implemented here; no original TransUNet package is vendored. The same distinction applies to the independent adaptations of AAU-Net, HyperFusion, and MAP. MIT does not relicense external author repositories. Obtain external code separately if an experiment requires those authors' original implementations.

## Data and figures

Datasets, annotations, clinical records, source manuscript PDF, case-level results, and model weights are not distributed. `assets/framework.png` is a schematic drawn from the method description and contains no patient images. Dataset access and reuse remain governed by the respective providers.
