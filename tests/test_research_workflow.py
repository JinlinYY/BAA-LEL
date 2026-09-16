"""Synthetic checks for folds, preprocessing, checkpoint export, and objectives."""
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from bua_lel.config import load_config
from bua_lel.engine import trainer
from bua_lel.engine.inference import clinical_tensor
from bua_lel.engine.losses import make_nomissing_loss_fn
from bua_lel.data.dataset import DualTaskDataset


class SmallJointModel(nn.Module):
    def __init__(self,clinical_dim,classes):
        super().__init__()
        self.encoder=nn.Module()
        self.encoder.image_encoder=nn.Conv2d(3,2,1)
        self.seg_head=nn.Conv2d(2,1,1)
        self.cls_head=nn.Linear(2+clinical_dim,classes)
    def forward(self,image,c_obs,m,task='both'):
        feature=self.encoder.image_encoder(image)
        seg=self.seg_head(feature)
        cls=self.cls_head(torch.cat([feature.mean((2,3)),c_obs],1))
        return seg,cls,{'seg_logits_low_raw':seg,'boundary_anchor_loss':seg.sum()*0}


def test_all_research_configurations():
    root=Path(__file__).resolve().parents[1]
    files=[p for p in (root/'configs').rglob('*.yaml') if p.name!='base.yaml' and p.parent.name!='nested']
    for path in files:
        cfg=load_config(path)
        assert cfg.image_size==256
        assert cfg.folds==5


def test_segmentation_objective_has_no_classification_gradient():
    model=SmallJointModel(0,3)
    loss_fn=make_nomissing_loss_fn(nn.CrossEntropyLoss(),0.5,True,segmentation_only=True)
    x=torch.rand(2,3,8,8);c=torch.empty(2,0)
    loss,_=loss_fn(model=model,x_img=x,seg_gt=torch.zeros(2,1,8,8),has_mask=torch.ones(2),y_gt=torch.tensor([0,2]),c_obs=c,m=c)
    loss.backward()
    assert model.seg_head.weight.grad.abs().sum()>0
    assert model.cls_head.weight.grad is None or model.cls_head.weight.grad.abs().sum()==0


def test_clinical_transform_matches_fold_statistics():
    payload={'feature_names':['Age','ER__negative','ER__positive'],'numeric_slice':(0,1),'num_scaler':{'mean':[50.0],'std':[10.0]},'cat_maps':{'ER':{'negative':0,'positive':1}},'onehot_slices':{'ER':(1,3)},'clinical_dim':3}
    torch.testing.assert_close(clinical_tensor(payload,{'Age':60,'ER':'positive'}),torch.tensor([[1.,0.,1.]]))


def test_fivefold_training_and_checkpoint_export(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    images=tmp_path/'images';masks=tmp_path/'masks';images.mkdir();masks.mkdir()
    rows=[]
    for i in range(20):
        pid=f'p{i:03d}'
        image=np.full((16,16),30+i,np.uint8)
        mask=np.zeros((16,16),np.uint8);mask[4:12,4:12]=255
        cv2.imwrite(str(images/f'{pid}.png'),image);cv2.imwrite(str(masks/f'{pid}.png'),mask)
        rows.append({'patient_id':pid,'Age':40+i,'label':i%2})
    table=tmp_path/'clinical.csv';pd.DataFrame(rows).to_csv(table,index=False)
    cfg=trainer.TrainConfig(image_dir=str(images),mask_dir=str(masks),clinical_excel=str(table),save_dir=str(tmp_path/'output'),fold_manifest_path=str(tmp_path/'folds.json'),benchmark_output_root=str(tmp_path/'comparison'),device='cpu',folds=5,epochs=1,batch_size=2,num_workers=0,num_classes=2,image_size=16,freeze_medsam=False,selection_policy='fixed_final_epoch',save_final_oof_roc=False,save_oof_npz=False,experiment_name='synthetic_cohort')
    monkeypatch.setattr(trainer,'build_model',lambda cfg,ds:SmallJointModel(ds.get_feature_dim(),cfg.num_classes))
    trainer.main(cfg)
    frame=pd.read_csv(tmp_path/'output/oof_cases.csv')
    assert len(frame)==20 and frame.pid.is_unique and set(frame.fold)=={1,2,3,4,5}
    state=torch.load(tmp_path/'output/best_fold1.pth',weights_only=False)
    assert state['clinical_dim']==1 and state['feature_names']==['Age']
    assert state['num_scaler'] is not None


def test_image_only_dataset_has_zero_clinical_width(tmp_path):
    images=tmp_path/'images';images.mkdir()
    cv2.imwrite(str(images/'p0.png'),np.zeros((8,8),np.uint8))
    table=tmp_path/'clinical.csv';pd.DataFrame({'patient_id':['p0'],'label':[0]}).to_csv(table,index=False)
    dataset=DualTaskDataset(str(images),None,str(table),mode='cls',allow_missing_mask=True)
    assert dataset.get_feature_dim()==0


def test_complete_observed_inference_rejects_missing_values():
    state={'clinical_dim':1,'feature_names':['Age'],'numeric_slice':(0,1),'num_scaler':{'mean':[50.],'std':[10.]},'cfg':{'clinical_preprocessing_profile':'complete_observed'}}
    with pytest.raises(ValueError,match='requires values'):
        clinical_tensor(state,{})


def test_binary_one_mask_is_foreground(tmp_path):
    path=tmp_path/'mask.png'
    cv2.imwrite(str(path),np.ones((8,8),np.uint8))
    ds=SimpleNamespace(H=8,W=8)
    mask=DualTaskDataset._read_mask(ds,str(path))
    assert mask.sum()==64


def test_inference_reconstructs_a_fold_checkpoint(tmp_path,monkeypatch):
    from dataclasses import asdict
    from bua_lel.engine import inference
    cfg=trainer.TrainConfig(num_classes=2)
    model=SmallJointModel(1,2).eval()
    path=tmp_path/'fold.pth'
    torch.save({'cfg':asdict(cfg),'clinical_dim':1,'numeric_slice':(0,1),'onehot_slices':{},'feature_names':['Age'],'num_scaler':{'mean':[50.],'std':[10.]},'cat_maps':{},'model_state':model.state_dict()},path)
    monkeypatch.setattr(inference,'build_model',lambda cfg,schema:SmallJointModel(schema.get_feature_dim(),cfg.num_classes))
    restored,_,state=inference.load_model(path)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,restored.state_dict()[key])
    assert state['clinical_dim']==1
