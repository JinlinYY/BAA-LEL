# Experimental protocols and commands

## Primary configuration

`configs/base.yaml` specifies 100 epochs, batch size 4, AdamW with learning rate and weight decay 1e-4, cosine decay, gradient clipping 1.0, patience 12, 64 boundary nodes, 128 graph hidden units, three graph layers, displacement bound 0.08, and anchor coefficient 0.01. MedSAM is frozen except for its last Transformer block. Classification/fusion parameter groups use twice the base learning rate.

The default selection score is `0.3 * Dice + 0.7 * Macro-F1`; this is an implementation choice, not an equation specified in the manuscript. The main trainer's `best_composite_early_stop` policy selects on the held-out fold and evaluates the selected checkpoint on that same fold. These are **fold-validation estimates**, not an independent nested test estimate. `fixed_final_epoch` performs no checkpoint selection on the held-out fold. The baseline runners instead create inner validation subsets from outer training patients and evaluate the outer fold once. State the chosen protocol when reporting results.

Five-fold means and sample standard deviations are reported separately from pooled case metrics. A single selected fold does not estimate cross-fold variation; the main trainer's single-fold SD fields are zero placeholders. HD95 and ASSD are measured in pixels on the evaluation grid, not millimeters. Empty-mask conventions are defined in `baselines/metrics.py` and covered by tests.

The dataset configurations expose the model and tasks for further research. Private cohorts, fixed patient manifests, full training histories, and trained weights are absent, so exact numerical agreement with manuscript tables is not certified. In particular, LMNUSC and BrEaST cohort-specific source experiments include different learning rates, batching, and classifier widths from the common settings. Use the nested cohort configurations when studying those settings; do not present the common settings as a reconstruction of every table entry.

## Cross-validation

```bash
python scripts/cross_validate.py --config configs/her2usc.yaml
python scripts/cross_validate.py --config configs/lmnusc.yaml
python scripts/cross_validate.py --config configs/breast.yaml
python scripts/cross_validate.py --config configs/busi.yaml
python scripts/cross_validate.py --config configs/imaplusplus.yaml
python scripts/cross_validate.py --config configs/imaplusplus_multiclass.yaml
```

`--fold 1 --fold 2` limits execution to selected folds. `--output-dir` selects a fresh directory. Completed predictions, split manifests, configuration records, and fold preprocessing should remain together. Existing nonempty output directories are rejected by the primary command.

The IMA++ configurations preserve the dataset-specific source runner: ImageNet ResNet-50 with its last bottleneck trainable, RGB 256×256 inputs, 30 epochs, batch size 8, patience 8, mild dermoscopy augmentation, 0.4 BCE + 0.6 Dice, and equal Dice/F1 checkpoint-selection weights. This differs from the MedSAM settings in the manuscript's common experimental description. Binary and three-class endpoints are distinct experiments.

## Held-out checkpoint evaluation

For nested patient-level evaluation with cohort-specific settings, use:

```bash
python scripts/nested_cross_validate.py --config configs/nested/her2usc.yaml --data-root /path/to/source
python scripts/nested_cross_validate.py --config configs/nested/lmnusc.yaml --data-root /path/to/source
python scripts/nested_cross_validate.py --config configs/nested/breast.yaml --data-root /path/to/source
```

These configurations use the baseline adapters' source layout. LMNUSC declares its categorical clinical schema explicitly; prepare a table with matching categories and consistent percentage units. Its configuration uses 120 epochs, learning rate 5e-5, effective batch size 4, and a 128-dimensional classifier projection. BrEaST uses learning rate 7e-5, a 512-dimensional projection, and excludes Diagnosis/Verification. `configs/nested/busi_joint.yaml` retains a separate image-only **joint** experiment from the available implementation; it trains a diagnostic head as well as segmentation and is distinct from the segmentation-only BUSI configuration.

The single-fold evaluation and inference commands below load checkpoints from the **primary trainer**. Nested baseline checkpoints have a different serialization schema and are evaluated by their own runner; do not pass them to `scripts/inference.py`.

```bash
python scripts/evaluate.py \
  --checkpoint outputs/HER2USC/bua_lel/best_fold1.pth \
  --image-dir data/HER2USC/images --mask-dir data/HER2USC/masks \
  --clinical-table data/HER2USC/clinical.csv \
  --ids /path/to/held_out_ids.txt --output-dir outputs/evaluation --device cuda
```

The identifier file contains one case per line. Select only the corresponding fold's held-out cases or an independent cohort. The evaluator uses the checkpoint's preprocessing without fitting on evaluation data and saves `cases.csv` and `metrics.json`. It cannot establish patient independence for an arbitrary external identifier list.

## Ablations and sensitivity

```bash
python scripts/run_ablations.py --config configs/ablations/no_anchor_graph.yaml
python scripts/run_ablations.py --config configs/ablations/no_zonal_morphology.yaml
python scripts/run_ablations.py --dry-run
python analysis/roi_sensitivity.py --config configs/her2usc.yaml \
  --boundary 0.10 0.15 0.20 --peritumoral 0.30 0.40 0.50 \
  --output-dir outputs/roi_sensitivity
```

Other configurations cover boundary ambiguity, geometry/ambiguity evidence, each zone, fusion strategies, modalities, encoder tuning, SAM ViT-B, ImageNet ResNet-50, and MedSAM-MTL. They inherit the same fold-manifest path. SAM requires its own checkpoint. ResNet-50 uses torchvision ImageNet weights unless a local checkpoint is supplied.

## Calibration

```bash
python analysis/calibration.py --task classification \
  --predictions outputs/HER2USC/bua_lel/oof_predictions.npz \
  --output-dir outputs/calibration
```

Classification uses top-label confidence for ECE and the summed multiclass Brier score. Segmentation input is an NPZ with `probabilities` and binary `targets` arrays of the same shape; its ECE is foreground-probability calibration. The streaming accumulator in `bua_lel/utils/calibration.py` supports large pixel collections. Neither procedure applies temperature fitting. Any learned recalibration must use a separate calibration partition.

## Paired statistical comparisons

```bash
python analysis/statistical_comparison.py --method outputs/bua_lel/cases.csv \
  --baseline outputs/miinet/cases.csv --baseline outputs/mtanet/cases.csv \
  --metrics ACC Macro-F1 Macro-AUC --resamples 10000 \
  --output outputs/statistical_comparison.csv
```

Input files require identical unique `pid` values and `fold` assignments. Classification additionally requires `y_true` and `proba_0`, `proba_1`, etc. To analyze the primary trainer's `oof_cases.csv`, rename its `prob_c0`, `prob_c1`, etc. columns to these names. Segmentation files require Dice, mIoU, HD95, and/or ASSD. ACC uses the exact McNemar test; other metrics use paired permutations. Confidence intervals use paired bootstrap sampling, class-stratified for classification. Holm correction covers **all comparisons passed in the same invocation**. Include the entire intended hypothesis family in one run.

## Evidence and efficiency

```bash
python analysis/interpretability.py --checkpoint /path/to/best_fold1.pth \
  --image /path/to/image.png --clinical-json /path/to/variables.json \
  --output-dir outputs/evidence --device cuda
python analysis/model_efficiency.py --checkpoint /path/to/best_fold1.pth \
  --device cuda --warmup 5 --iterations 30 --output outputs/efficiency.json
```

Evidence export includes coarse/refined lesion probabilities, boundary ambiguity, boundary responses, and geometry/ambiguity embeddings. These visualizations describe internal representations and should not be interpreted as causal explanations. Efficiency measurements report float32 forward latency on synthetic tensors, with device synchronization and warmup; file loading and image preprocessing are excluded.

Clinical-variable permutation on a held-out fold is available with:

```bash
python analysis/clinical_permutation.py --checkpoint /path/to/best_fold1.pth \
  --image-dir /path/to/images --clinical-table /path/to/clinical.csv \
  --ids /path/to/held_out_ids.txt --repeats 20 \
  --output-dir outputs/clinical_permutation --device cuda
```

Each repetition permutes one raw clinical column across held-out cases before applying the checkpoint's fixed transformation. Reported differences are baseline minus perturbed ACC, Macro-F1, and Macro-AUC. Positive values indicate performance reduction. This is marginal permutation importance; correlations between variables can affect its interpretation.
