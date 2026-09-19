"""Validated YAML configuration for patient-level experiments."""
from dataclasses import asdict, fields
from pathlib import Path
import yaml
from baa_lel.engine.trainer import TrainConfig, validate_train_config


def load_config(path, overrides=None):
    path = Path(path).resolve()
    def read(source, visited):
        if source in visited:
            raise ValueError('Cyclic configuration inheritance')
        values = yaml.safe_load(source.read_text(encoding='utf-8')) or {}
        parent = values.pop('extends', None)
        base = read((source.parent / parent).resolve(), visited | {source}) if parent else {}
        return {**base, **values}
    values = read(path, set())
    values.update(overrides or {})
    known = {item.name for item in fields(TrainConfig)}
    if set(values) - known:
        raise ValueError(f'Unknown configuration fields: {sorted(set(values) - known)}')
    for key in ('selected_folds', 'enabled_regions', 'clinical_exclude_columns', 'cls_name_keywords'):
        if key in values:
            values[key] = tuple(values[key])
    cfg = TrainConfig(**values)
    validate_train_config(cfg)
    if cfg.training_task == 'segmentation_only' and (cfg.score_w_seg != 1 or cfg.score_w_cls != 0):
        raise ValueError('Segmentation-only selection requires score_w_seg=1, score_w_cls=0')
    return cfg


def validate_inputs(cfg):
    from baa_lel.data.preprocessing import read_excel_df, build_pid_and_labels
    for value in (cfg.image_dir, cfg.mask_dir, cfg.clinical_excel):
        if not Path(value).exists():
            raise FileNotFoundError(value)
    if cfg.visual_backbone in {'medsam_vit_b', 'sam_vit_b'}:
        checkpoint = cfg.visual_checkpoint_path or cfg.medsam_checkpoint_path
        if not checkpoint or not Path(checkpoint).is_file():
            raise FileNotFoundError(f'Pretrained encoder checkpoint required: {checkpoint}')
    frame = read_excel_df(cfg.clinical_excel)
    if frame.iloc[:, 0].duplicated().any():
        raise ValueError('Case identifiers must be unique')
    pids, labels = build_pid_and_labels(cfg.image_dir, cfg.clinical_excel)
    if len(pids) != len(frame):
        raise ValueError('Every metadata row must have one matching image')
    if len(labels) == 0 or min(labels) < 0 or max(labels) >= cfg.num_classes:
        raise ValueError('Labels must be in [0, num_classes)')
    if cfg.training_task == 'joint' and len(set(labels)) != cfg.num_classes:
        raise ValueError('All configured classes must be represented')
    return {'cases': len(pids), 'classes': sorted(set(map(int, labels)))}
