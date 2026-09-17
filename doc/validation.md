# Testing

The test suite uses synthetic inputs to check model interfaces, data processing, training, and evaluation.

Checks include:

- Syntax parsing and `--help` execution of every research command.
- Dataset and ablation YAML parsing.
- A complete, randomly initialized MedSAM-based BUA-LEL forward pass at a 64×64 encoder input size, verifying segmentation/classification dimensions and finite boundary evidence.
- Segmentation baseline output dimensions and inductive HetMed query/reference handling.
- Clinical fusion, positional embedding interpolation, and activation checkpointing behavior.
- Five-fold training, prediction coverage, and fold-specific checkpoint/preprocessor export on a 20-case synthetic cohort with a small surrogate network.
- Segmentation-only loss gradients, binary masks, clinical preprocessing, and missing-value rejection for complete-observed inference.

The GitHub Actions workflow runs with Python 3.10 and CPU PyTorch 2.8.0.

Run locally from the repository root:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q tests
```
