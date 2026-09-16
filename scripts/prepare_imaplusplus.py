"""Prepare group-stratified IMA++ manifests for joint segmentation/classification."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd


CLINICAL_COLUMNS = [
    "age_approx",
    "sex",
    "anatom_site_general",
    "anatom_site_special",
    "clin_size_long_diam_mm",
    "fitzpatrick_skin_type",
    "family_hx_mm",
    "personal_hx_mm",
    "dermoscopic_type",
    "image_type",
]


def _group_id(row: pd.Series) -> str:
    patient = str(row.get("patient_id", "")).strip()
    if patient and patient.lower() != "nan":
        return f"patient:{patient}"
    lesion = str(row.get("lesion_id", "")).strip()
    if lesion and lesion.lower() != "nan":
        return f"lesion:{lesion}"
    return f"image:{row['isic_id']}"


def _select_masks(seg: pd.DataFrame) -> pd.DataFrame:
    skill_rank = {"ST": 0, "S1": 1, "S2": 2, "MV": 3}
    tool_rank = {"ST": 0, "T1": 1, "T2": 2, "T3": 3, "MV": 4}
    work = seg.copy()
    work["_skill_rank"] = work["skill_level"].map(skill_rank).fillna(9)
    work["_tool_rank"] = work["tool"].map(tool_rank).fillna(9)
    work = work.sort_values(
        ["ISIC_id", "_skill_rank", "_tool_rank", "annotator", "seg_filename"]
    )
    selected = work.groupby("ISIC_id", as_index=False).first()
    return selected.drop(columns=["_skill_rank", "_tool_rank"])


def _link_or_copy(source: Path, destination: Path) -> str:
    if destination.exists():
        return "existing"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def prepare(root: Path) -> dict:
    downloads = root / "downloads"
    raw_masks = root / "masks"
    images = root / "images"
    metadata_dir = root / "metadata"
    selected_dir = root / "selected_masks"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)

    image_meta = pd.read_csv(downloads / "img_metadata.csv")
    seg_meta = pd.read_csv(downloads / "seg_metadata.csv")
    selected = _select_masks(seg_meta)

    link_counts = {"existing": 0, "hardlink": 0, "copy": 0}
    missing_masks = []
    for row in selected.itertuples(index=False):
        source = raw_masks / row.seg_filename
        destination = selected_dir / f"{row.ISIC_id}.png"
        if not source.exists():
            missing_masks.append(str(source))
            continue
        link_counts[_link_or_copy(source, destination)] += 1
    if missing_masks:
        raise FileNotFoundError(f"missing {len(missing_masks)} selected masks; first={missing_masks[0]}")

    selected[[
        "ISIC_id", "seg_filename", "annotator", "tool", "skill_level", "mask_md5"
    ]].to_csv(metadata_dir / "selected_masks.csv", index=False)

    frame = image_meta.rename(columns={"ISIC_id": "isic_id"}).copy()
    frame["group_id"] = frame.apply(_group_id, axis=1)
    keep = ["isic_id", *CLINICAL_COLUMNS, "group_id"]

    binary = frame[frame["benign_malignant"].isin(["benign", "malignant"])].copy()
    binary["label"] = binary["benign_malignant"].map({"benign": 0, "malignant": 1})
    binary[keep + ["label"]].to_csv(metadata_dir / "clinical_binary.csv", index=False)

    multiclass = frame[frame["diagnosis_1"].isin(["Benign", "Malignant", "Indeterminate"])].copy()
    multiclass["label"] = multiclass["diagnosis_1"].map(
        {"Benign": 0, "Malignant": 1, "Indeterminate": 2}
    )
    multiclass[keep + ["label"]].to_csv(
        metadata_dir / "clinical_multiclass.csv", index=False
    )

    image_stems = {
        path.stem for path in images.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    selected_ids = set(selected["ISIC_id"].astype(str))
    report = {
        "version": 1,
        "root": str(root.resolve()),
        "image_metadata_rows": int(len(image_meta)),
        "segmentation_metadata_rows": int(len(seg_meta)),
        "selected_mask_count": int(len(selected)),
        "selected_mask_policy": "STAPLE consensus when available; otherwise S1 before S2 and T1 before T2 before T3",
        "selected_mask_materialization": link_counts,
        "downloaded_image_count": int(len(image_stems)),
        "downloaded_images_with_selected_mask": int(len(image_stems & selected_ids)),
        "binary_rows": int(len(binary)),
        "binary_distribution": {
            str(key): int(value) for key, value in binary["label"].value_counts().sort_index().items()
        },
        "multiclass_rows": int(len(multiclass)),
        "multiclass_classes": {"0": "Benign", "1": "Malignant", "2": "Indeterminate"},
        "multiclass_distribution": {
            str(key): int(value)
            for key, value in multiclass["label"].value_counts().sort_index().items()
        },
        "clinical_columns": CLINICAL_COLUMNS,
        "group_policy": "patient_id, else lesion_id, else isic_id",
    }
    (metadata_dir / "preparation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path("data/IMAplusplus"),
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
