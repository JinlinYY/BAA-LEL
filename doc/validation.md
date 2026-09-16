# Validation scope

The public package passed 30 local tests with synthetic data; private cohorts and trained model weights are not part of these checks.

Checks include:

- Syntax parsing and `--help` execution of every research command.
- Dataset and ablation YAML parsing.
- A complete, randomly initialized MedSAM-based BUA-LEL forward pass at a 64×64 encoder input size, verifying segmentation/classification dimensions and finite boundary evidence.
- Segmentation baseline output dimensions and inductive HetMed query/reference handling.
- Clinical fusion, positional embedding interpolation, and activation checkpointing behavior.
- Five-fold training, prediction coverage, and fold-specific checkpoint/preprocessor export on a 20-case synthetic cohort with a small surrogate network.
- Segmentation-only loss gradients, binary masks, clinical preprocessing, and missing-value rejection for complete-observed inference.

Local execution used Windows, Python **3.9.25**, PyTorch **2.8.0+cu128**, torchvision **0.23.0**, and torch-geometric **2.6.1**. These were direct source tests using an available environment; they do not constitute installation validation of the declared Python >=3.10 package. The included GitHub Actions workflow targets Python 3.10 and CPU PyTorch 2.8.0. Its status is independent of the local checks.

The manuscript's Python 3.10 / PyTorch 2.0.1 / CUDA 11.8 stack, full-resolution training with pretrained weights, CUDA memory/latency measurements, and manuscript numerical results have **not** been revalidated by these synthetic checks. The small-input forward check verifies implementation wiring; it is not a performance experiment.

Run locally from the repository root:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q tests
```
