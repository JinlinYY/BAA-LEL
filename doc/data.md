# Data preparation

No images, masks, clinical tables, case identifiers, split files, or model weights are included. Obtain each dataset under its provider's terms. HER2USC and LMNUSC are private clinical cohorts; this repository does not grant access to them.

## Common training format

```text
data/HER2USC/
├── images/case_000001.png
├── masks/case_000001.png
└── clinical.csv
```

The clinical table can be CSV, TSV, or XLSX. Its **first column is the case identifier**, its **last column is the integer class label**, and intermediate columns are clinical variables. Image and mask stems must match the identifier exactly. A purely synthetic schema example is:

```text
patient_id,Age,ER,PR,Ki-67%,label
example_001,50,positive,negative,20,0
```

Use numeric columns for continuous measurements and string columns for categories. Keep column order and units fixed across training and inference. Do not mix percentages represented as 0–1 fractions with 0–100 percentages. Standardization and category maps are fitted within training folds. The `complete_observed` profile requires complete clinical data and categories represented in the training fold; `missing_aware` uses training-fold mean imputation plus explicit missing/unknown categories.

Masks are binary: zero denotes background and positive values denote lesion foreground. Images use bilinear resizing; masks use nearest-neighbor resizing. When multiple source masks describe a lesion, their union is used. Missing masks are excluded from segmentation loss and metrics. Never substitute a zero mask for an unavailable annotation; a true empty mask is valid background supervision.

Identifiers in published datasets can be image-level identifiers. Where multiple images belong to a patient or lesion, supply a `group_id`, exclude it from clinical features, and select `fold_manifest_strategy: stratified_group` with `fold_group_column: group_id`. Group identity must be shared by every image from that patient/lesion.

## Cohort adapters

```bash
python scripts/prepare_data.py --dataset HER2USC --source-root /path/to/source --output-root data
python scripts/prepare_data.py --dataset BrEaST --source-root /path/to/source --output-root data
python scripts/prepare_data.py --dataset BUSI --source-root /path/to/source --output-root data
```

The source-root layouts used by these adapters and the baseline runners are:

- HER2USC: `HER2/images/`, `HER2/masks/`, `HER2/clinical.xlsx`.
- LMNUSC: `LNM/images/`, `LNM/masks/`, `LNM/clinical.xlsx`.
- BrEaST: `BrEaST/images and masks/` and `BrEaST/clinical.xlsx`, with `CaseID`, `Image_filename`, `Mask_tumor_filename`, and `Classification` columns.
- BUSI: `BUSI/benign/`, `BUSI/malignant/`, `BUSI/normal/`, with source `_mask` files.

Preparation creates deterministic case identifiers and a private `source_mapping.csv`. Do not publish this mapping. Normalized layouts used by the main trainer and source layouts used by baseline adapters are different; pass the corresponding root to each command. For paired comparisons, use identical case identities and fold assignments for every method. Renaming identifiers or independently regenerating folds can change the patient split even with the same random seed.

HER2USC uses `0=HER2-zero`, `1=HER2-low`, `2=HER2-positive`. LMNUSC uses `0=LN0`, `1=LN1–3`, `2=LN≥4`. BrEaST uses `0=non-malignant (benign + normal)`, `1=malignant`; diagnosis, verification, filenames, and the target are excluded from clinical features. BUSI retains the benign/malignant/normal codes for stratified image splitting, while `training_task: segmentation_only` prevents these labels from entering the optimized objective. BUSI lacks a guaranteed patient identifier in this adapter; describe its folds as image-level unless an external grouping is provided.

## IMA++

```text
data/IMAplusplus/
├── downloads/img_metadata.csv
├── downloads/seg_metadata.csv
├── images/<isic_id>.jpg
└── masks/<source-mask-name>.png
```

Run `python scripts/prepare_imaplusplus.py --root data/IMAplusplus`. It selects one annotation per image using the declared consensus/annotator ordering, writes `selected_masks/`, and creates `metadata/clinical_binary.csv` and `metadata/clinical_multiclass.csv`. Binary labels are benign/malignant; three-class labels are Benign/Malignant/Indeterminate. The metadata retains `group_id` for grouped splitting and excludes it from clinical features. Available patient identity takes priority over lesion identity, with image identity as the final fallback.

## ISIC2018 segmentation

Provide `data/ISIC2018/images/`, `data/ISIC2018/masks/`, and a two-column `clinical.csv` containing `patient_id,label`, with label 0 for every image. Rename source `<id>_segmentation.png` masks to `<id>.png`. This configuration uses images and segmentation masks without tabular clinical inputs.
