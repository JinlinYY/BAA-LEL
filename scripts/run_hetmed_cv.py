"""Formal, inductive HetMed runner for HER2USC, LMNUSC and BrEaST."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.data import ClinicalPreprocessor, load_clinical_frame
from baselines.metrics import classification_metrics
from baselines.models import MODEL_METADATA, build_model
from baselines.models.classification import HetMedImageEncoder, HetMedLoss, InductiveHetMedGraphBuilder
from baselines.protocol import (
    PROTOCOL_VERSION,
    class_names,
    create_or_load_inner_manifest,
    create_or_load_patient_manifest,
    protocol_records,
    records_for_pids,
)
from baselines.results import save_per_class_metrics
from baselines.runner import seed_everything


@torch.no_grad()
def extract_features(records, image_size, checkpoint, backbone, output_dim, pretrained, device, batch_size, num_workers):
    from baselines.data import BaselineDataset
    from torch.utils.data import DataLoader

    data = BaselineDataset(records, np.zeros((len(records), 0), np.float32), image_size, augment=False, grayscale=records[0].image_path.parents[1].name != "IMAplusplus")
    worker_args = {"persistent_workers": True, "prefetch_factor": 8} if num_workers > 0 else {}
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"), **worker_args)
    encoder = HetMedImageEncoder(backbone=backbone, output_dim=output_dim, checkpoint_path=checkpoint, pretrained=pretrained).to(device).eval()
    features, pids = [], []
    for batch in loader:
        features.append(encoder(batch["image"].to(device)).cpu())
        pids.extend(map(str, batch["pid"]))
    return {pid: feature for pid, feature in zip(pids, torch.cat(features))}


def _evaluate(model, query_pids, reference_pids, feature_by_pid, query_clinical, reference_clinical, graph, labels, device, fold=None):
    model.eval()
    with torch.no_grad():
        output = model(
            torch.stack([feature_by_pid[pid] for pid in query_pids]).to(device),
            torch.as_tensor(query_clinical, device=device),
            torch.stack([feature_by_pid[pid] for pid in reference_pids]).to(device),
            torch.as_tensor(reference_clinical, device=device), graph.to(device),
        )
        probabilities = torch.softmax(output["logits"], 1).cpu().numpy()
    rows = []
    for pid, label, probability in zip(query_pids, labels, probabilities):
        row = {"pid": str(pid), "y_true": int(label), "y_pred": int(probability.argmax())}
        row.update({f"proba_{index}": float(value) for index, value in enumerate(probability)})
        if fold is not None:
            row["fold"] = int(fold)
        rows.append(row)
    return rows


def _score(rows, names):
    frame = pd.DataFrame(rows)
    return float(classification_metrics(frame["y_true"], frame[[f"proba_{name}" for name in names]].to_numpy(), names)["Macro-F1"])


def _graph_thresholds(args):
    if args.graph_thresholds == "auto":
        return None
    values = tuple(float(value.strip()) for value in args.graph_thresholds.split(",") if value.strip())
    if len(values) == 1:
        return values[0]
    if len(values) != args.num_relations:
        raise ValueError("--graph-thresholds needs one value or one value per relation")
    return values


def _author_grid(args):
    if args.no_author_grid:
        return [(args.lr, args.supervised_coefficient)]
    return [(lr, beta) for lr in (1e-4, 5e-4, 1e-3) for beta in (.01, .1, 1.0)]


def _class_weights(labels, num_classes, device):
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights /= weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _fit_epoch_count(train_pids, dev_pids, clinical, feature_by_pid, labels_by_pid, args, device, fold, names, fold_dir):
    preprocessor = ClinicalPreprocessor().fit(clinical.loc[train_pids])
    train_clinical = preprocessor.transform(clinical.loc[train_pids])
    dev_clinical = preprocessor.transform(clinical.loc[dev_pids])
    graph_builder = InductiveHetMedGraphBuilder(args.num_relations, _graph_thresholds(args), args.seed + fold).fit(train_clinical)
    train_graph = graph_builder.training_graph(train_clinical)
    dev_graph = graph_builder.query_graph(dev_clinical, train_clinical)
    train_image = torch.stack([feature_by_pid[pid] for pid in train_pids]).to(device)
    train_clinical_tensor = torch.as_tensor(train_clinical, device=device)
    train_labels = torch.as_tensor([labels_by_pid[pid] for pid in train_pids], device=device)
    class_weights = _class_weights(train_labels.detach().cpu().numpy(), len(names), device)
    history, best_selection = [], None
    for trial, (learning_rate, supervised_coefficient) in enumerate(_author_grid(args), start=1):
        seed_everything(args.seed + fold * 100 + trial)
        model = build_model(
            "hetmed", "classification", image_dim=next(iter(feature_by_pid.values())).numel(),
            clinical_dim=train_clinical.shape[1], num_classes=len(names),
            hidden_dim=args.hidden_dim, num_relations=args.num_relations,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
        criterion = HetMedLoss(supervised_coefficient, args.regularization_coefficient, class_weights)
        best_epoch, best_score, stale = 1, -np.inf, 0
        for epoch in range(1, args.epochs + 1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            output = model(train_image, train_clinical_tensor, train_image, train_clinical_tensor, train_graph.to(device))
            loss = criterion(output, train_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            rows = _evaluate(model, dev_pids, train_pids, feature_by_pid, dev_clinical, train_clinical, dev_graph, [labels_by_pid[pid] for pid in dev_pids], device)
            score = _score(rows, names)
            history.append({
                "fold": fold, "trial": trial, "learning_rate": learning_rate,
                "supervised_coefficient": supervised_coefficient, "epoch": epoch,
                "train_loss": float(loss.detach().cpu()), "development_score": score,
            })
            if score > best_score:
                best_epoch, best_score, stale = epoch, score, 0
            else:
                stale += 1
                if stale >= args.patience:
                    break
        candidate = {
            "fold": fold, "selected_epoch": best_epoch,
            "selection_metric": "Macro-F1", "selection_score": best_score,
            "learning_rate": learning_rate,
            "supervised_coefficient": supervised_coefficient,
            "hidden_dim": args.hidden_dim, "num_relations": args.num_relations,
            "graph_thresholds": args.graph_thresholds,
        }
        if best_selection is None or (candidate["selection_score"], -candidate["selected_epoch"]) > (best_selection["selection_score"], -best_selection["selected_epoch"]):
            best_selection = candidate
    pd.DataFrame(history).to_csv(fold_dir / "selection_metrics.csv", index=False, encoding="utf-8-sig")
    return best_selection


def _fit_final(train_pids, test_pids, clinical, feature_by_pid, labels_by_pid, args, device, fold, names, selection, fold_dir):
    seed_everything(args.seed + 1000 + fold)
    preprocessor = ClinicalPreprocessor().fit(clinical.loc[train_pids])
    train_clinical = preprocessor.transform(clinical.loc[train_pids])
    test_clinical = preprocessor.transform(clinical.loc[test_pids])
    graph_builder = InductiveHetMedGraphBuilder(args.num_relations, _graph_thresholds(args), args.seed + fold).fit(train_clinical)
    train_graph = graph_builder.training_graph(train_clinical)
    test_graph = graph_builder.query_graph(test_clinical, train_clinical)
    model = build_model("hetmed", "classification", image_dim=next(iter(feature_by_pid.values())).numel(), clinical_dim=train_clinical.shape[1], num_classes=len(names), hidden_dim=int(selection["hidden_dim"]), num_relations=args.num_relations).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(selection["learning_rate"]), weight_decay=args.weight_decay)
    train_image, train_clinical_tensor = torch.stack([feature_by_pid[pid] for pid in train_pids]).to(device), torch.as_tensor(train_clinical, device=device)
    train_labels = torch.as_tensor([labels_by_pid[pid] for pid in train_pids], device=device)
    class_weights = _class_weights(train_labels.detach().cpu().numpy(), len(names), device)
    criterion = HetMedLoss(float(selection["supervised_coefficient"]), args.regularization_coefficient, class_weights)
    for _ in range(int(selection["selected_epoch"])):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(train_image, train_clinical_tensor, train_image, train_clinical_tensor, train_graph.to(device)), train_labels)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    torch.save({"model_state": model.state_dict(), "selection": selection, "fold": fold, "graph_builder": graph_builder.state_dict()}, fold_dir / "model.pt")
    return _evaluate(model, test_pids, train_pids, feature_by_pid, test_clinical, train_clinical, test_graph, [labels_by_pid[pid] for pid in test_pids], device, fold)


def main():
    parser = argparse.ArgumentParser(description="Formal leakage-safe inductive HetMed five-fold CV")
    parser.add_argument("--dataset", required=True, choices=["HER2USC", "LMNUSC", "BrEaST", "IMAplusplus"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="outputs/baselines")
    parser.add_argument("--image-checkpoint", help="Optional local ResNet/SimCLR checkpoint; otherwise the declared ImageNet adapter is used")
    parser.add_argument("--no-image-pretrained", action="store_true")
    parser.add_argument("--image-backbone", choices=["resnet18", "resnet50"], default="resnet50")
    parser.add_argument("--image-feature-dim", type=int, default=2048)
    parser.add_argument("--num-relations", type=int, default=4)
    parser.add_argument(
        "--graph-thresholds", default="0.9,0.9,0.9,0.75",
        help="Comma-separated per-relation thresholds; defaults to the authors' CMMD setting, or use auto",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--supervised-coefficient", type=float, default=.01, help="Used only with --no-author-grid")
    parser.add_argument("--regularization-coefficient", type=float, default=.001)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4, help="Used only with --no-author-grid")
    parser.add_argument("--no-author-grid", action="store_true", help="Disable the paper-range inner-fold lr/beta search")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=256, choices=[256])
    parser.add_argument("--inner-validation-fraction", type=float, default=.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    records = protocol_records(args.dataset, args.data_root)
    counts = np.bincount(np.asarray([record.label for record in records], dtype=np.int64))
    if counts[counts > 0].min() < 5:
        raise ValueError(f"each class needs five cases; counts={counts.tolist()}")
    outer = create_or_load_patient_manifest(args.output_root, args.dataset, records, seed=args.seed)
    inner = create_or_load_inner_manifest(args.output_root, args.dataset, records, outer, seed=args.seed, validation_fraction=args.inner_validation_fraction)
    clinical = load_clinical_frame(args.dataset, Path(args.data_root), records)
    if clinical.shape[1] == 0:
        raise ValueError("HetMed requires clinical variables")
    names, labels_by_pid = class_names(records), {record.pid: record.label for record in records}
    feature_by_pid = extract_features(records, args.image_size, args.image_checkpoint, args.image_backbone, args.image_feature_dim, not args.no_image_pretrained, device, args.batch_size, args.num_workers)
    root = Path(args.output_root) / args.dataset / "classification" / "hetmed"
    root.mkdir(parents=True, exist_ok=True)
    inner_by_fold = {int(entry["fold"]): entry for entry in inner["folds"]}
    all_rows, fold_rows = [], []
    for outer_fold in outer["folds"]:
        fold, fold_dir = int(outer_fold["fold"]), root / f"fold{outer_fold['fold']}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        oof_path = fold_dir / "oof_classification.csv"
        if args.resume and oof_path.exists():
            all_rows.extend(pd.read_csv(oof_path).to_dict("records"))
            fold_rows.append(json.loads((fold_dir / "selection.json").read_text(encoding="utf-8")))
            continue
        entry = inner_by_fold[fold]
        selection_path = fold_dir / "selection.json"
        if args.resume and selection_path.exists():
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
        else:
            selection = _fit_epoch_count(entry["train_pids"], entry["dev_pids"], clinical, feature_by_pid, labels_by_pid, args, device, fold, names, fold_dir)
            selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
        fold_rows.append(selection)
        rows = _fit_final(outer_fold["train_pids"], outer_fold["val_pids"], clinical, feature_by_pid, labels_by_pid, args, device, fold, names, selection, fold_dir)
        pd.DataFrame(rows).to_csv(oof_path, index=False, encoding="utf-8-sig")
        all_rows.extend(rows)
    oof = pd.DataFrame(all_rows).sort_values("pid")
    if oof["pid"].duplicated().any() or set(oof["pid"].astype(str)) != {record.pid for record in records}:
        raise RuntimeError("HetMed OOF coverage does not match the formal patient cohort")
    probability_columns = [f"proba_{name}" for name in names]
    metrics = classification_metrics(oof["y_true"], oof[probability_columns].to_numpy(), names)
    oof.to_csv(root / "oof_classification.csv", index=False, encoding="utf-8-sig")
    save_per_class_metrics(root / "per_class_metrics.csv", metrics["per_class"])
    flat = {key: value for key, value in metrics.items() if key != "per_class"}
    for name, values in metrics["per_class"].items():
        flat.update({f"class_{name}_{key}": value for key, value in values.items()})
    pd.DataFrame(fold_rows).to_csv(root / "selection_per_fold.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"fold": int(fold), **{key: value for key, value in classification_metrics(group["y_true"], group[probability_columns].to_numpy(), names).items() if key != "per_class"}}
        for fold, group in oof.groupby("fold")
    ]).to_csv(root / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    metadata = MODEL_METADATA["hetmed"]
    image_encoder = (
        str(args.image_checkpoint)
        if args.image_checkpoint
        else (
            f"ImageNet {args.image_backbone} fixed adapter"
            if not args.no_image_pretrained else f"torchvision {args.image_backbone} (untrained synthetic/inspection)"
        )
    )
    summary = {"dataset": args.dataset, "task": "classification", "method": "hetmed", "display_name": metadata["display_name"], "implementation": metadata["implementation"], "source": metadata["source"], "version": metadata["version"], "upstream_version": metadata["upstream_version"], "interactive": False, "eligible_for_automatic_comparison": True, "protocol_version": PROTOCOL_VERSION, "fold_fingerprint": outer["split_fingerprint"], "inner_fold_fingerprint": inner["split_fingerprint"], "image_encoder": image_encoder, "selection_metric": "Macro-F1", "graph_thresholds": args.graph_thresholds, **{key: round(float(value), 3) for key, value in flat.items()}}
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "run_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
