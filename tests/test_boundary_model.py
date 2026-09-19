"""Numerical checks for the complete uninitialized model and boundary evidence."""
import torch
from baa_lel.models import BAALEL
from baa_lel.models.boundary.anchor_boundary_graph import AnchorConstrainedBoundaryGraph


def test_complete_model_forward_without_external_weights():
    torch.set_num_threads(1)
    model=BAALEL(clinical_dim=2,numeric_slice=(0,2),onehot_slices_dict={},num_classes=3,image_size=64,medsam_checkpoint_path=None,freeze_medsam=True,unfreeze_medsam_last_n=0,late_raw_clinical_fusion=True).eval()
    with torch.no_grad():
        seg,cls,aux=model(torch.rand(1,3,64,64),c_obs=torch.rand(1,2),m=torch.ones(1,2),task='both')
    assert seg.shape==(1,1,64,64) and cls.shape==(1,3)
    assert torch.isfinite(seg).all() and torch.isfinite(cls).all()
    assert aux['z_geo'].shape==(1,256) and aux['z_unc'].shape==(1,256)
    assert aux['boundary_anchor_loss']>=0
