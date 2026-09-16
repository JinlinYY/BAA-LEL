"""Checkpoint reconstruction and fold-specific clinical transformation."""
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pandas as pd
import torch
from bua_lel.engine.trainer import TrainConfig, build_model


def load_model(checkpoint, device='cpu'):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = TrainConfig(**payload['cfg'])
    cfg.medsam_checkpoint_path = None
    cfg.visual_checkpoint_path = '' if cfg.visual_backbone == 'imagenet_resnet50' else None
    schema = SimpleNamespace(
        get_feature_dim=lambda: int(payload['clinical_dim']),
        numeric_slice=payload.get('numeric_slice'),
        onehot_slices=payload.get('onehot_slices') or {},
    )
    model = build_model(cfg, schema)
    model.load_state_dict(payload['model_state'], strict=True)
    return model.to(device).eval(), cfg, payload


def clinical_tensor(payload, values):
    names = payload.get('feature_names') or []
    numeric_count = (payload.get('numeric_slice') or (0,0))[1]
    numeric_names = names[:numeric_count]
    required = [*numeric_names, *(payload.get('onehot_slices') or {})]
    if payload.get('cfg', {}).get('clinical_preprocessing_profile') == 'complete_observed':
        missing = [name for name in required if name not in values or values[name] is None or pd.isna(values[name])]
        if missing:
            raise ValueError(f'Complete-observed checkpoint requires values for: {missing}')
    scaler = payload.get('num_scaler')
    vector=[]
    for index,name in enumerate(numeric_names):
        mean=float(scaler['mean'][index]) if scaler is not None else 0.0
        std=float(scaler['std'][index]) if scaler is not None else 1.0
        value=values.get(name, mean)
        value=mean if value is None or pd.isna(value) else float(value)
        vector.append((value-mean)/(std if std>=1e-6 else 1.0))
    for name,limits in sorted((payload.get('onehot_slices') or {}).items(), key=lambda x:x[1][0]):
        mapping=payload['cat_maps'][name]
        value=values.get(name)
        key='<MISSING>' if value is None or pd.isna(value) else str(value)
        index=mapping.get(key, mapping.get('<UNK>'))
        if index is None:
            raise ValueError(f'Unknown category for {name}: {key}')
        onehot=[0.0]*len(mapping); onehot[index]=1.0; vector.extend(onehot)
    if len(vector)!=payload['clinical_dim']:
        raise ValueError('Clinical feature schema does not match checkpoint')
    return torch.tensor([vector],dtype=torch.float32)


def image_tensor(path, cfg):
    image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE if cfg.image_mode=='grayscale' else cv2.IMREAD_COLOR)
    if image is None: raise FileNotFoundError(path)
    original_shape=image.shape[:2]
    if cfg.image_mode=='grayscale':
        image=np.repeat(image[...,None],3,axis=2)
    else: image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image=cv2.resize(image,(cfg.image_size,cfg.image_size),interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(image.transpose(2,0,1).copy()).float()[None]/255.0,original_shape


@torch.no_grad()
def predict(model,cfg,payload,image,clinical,device):
    x,shape=image_tensor(image,cfg)
    c=clinical_tensor(payload,clinical).to(device)
    seg,cls,aux=model(x.to(device),c_obs=c,m=torch.ones_like(c),task='both')
    return torch.sigmoid(seg)[0,0].cpu().numpy(),torch.softmax(cls,1)[0].cpu().numpy(),aux,shape
