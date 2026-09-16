"""Run configured cross-validation and export fold-level predictions."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bua_lel.config import load_config, validate_inputs
from bua_lel.engine.trainer import main as train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--fold', type=int, action='append', help='Repeat to select folds; default: all folds')
    parser.add_argument('--device')
    parser.add_argument('--output-dir')
    parser.add_argument('--dry-run', action='store_true', help='Validate configuration without loading data or weights')
    args = parser.parse_args()
    overrides = {}
    if args.fold: overrides['selected_folds'] = tuple(args.fold)
    if args.device: overrides['device'] = args.device
    if args.output_dir: overrides['save_dir'] = args.output_dir
    cfg = load_config(args.config, overrides)
    if args.dry_run:
        print(json.dumps(asdict(cfg), indent=2)); return
    print(json.dumps(validate_inputs(cfg), indent=2))
    out = Path(cfg.save_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Choose an empty output directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    (out/'config.json').write_text(json.dumps(asdict(cfg), indent=2), encoding='utf-8')
    train(cfg)
    if cfg.training_task == 'segmentation_only':
        # The shared tensor interface contains classification logits; they are not task results.
        import pandas as pd
        for path in out.rglob('*.csv'):
            if path.name in {'experiment_config.csv','fold_assignments.csv'}:
                continue
            frame=pd.read_csv(path)
            columns=[c for c in frame if c.startswith(('ACC','Macro-','prob_c','sens_','spec_')) or c in {'y_true','y_pred','acc','macro_f1','auc_macro_ovr','cls_loss','train_classification'}]
            frame=frame.drop(columns=columns)
            if 'metric' in frame:
                frame=frame[~frame.metric.str.startswith(('acc','macro_f1','auc_','sens_','spec_'))]
            frame.to_csv(path,index=False)


if __name__ == '__main__':
    main()
