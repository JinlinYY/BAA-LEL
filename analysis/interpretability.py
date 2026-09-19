"""Export coarse lesion priors, boundary ambiguity, and regional evidence."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from baa_lel.engine.inference import load_model,predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','image','output-dir'):p.add_argument('--'+key,required=True)
    p.add_argument('--clinical-json');p.add_argument('--device',default='cpu')
    a=p.parse_args();model,cfg,state=load_model(a.checkpoint,a.device)
    values=json.loads(Path(a.clinical_json).read_text()) if a.clinical_json else {}
    if state['clinical_dim'] and not a.clinical_json:p.error('--clinical-json is required')
    prob,classes,aux,_=predict(model,cfg,state,a.image,values,a.device)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    arrays={'refined_probability':prob}
    for key in ('seg_prob_low','boundary_uncertainty','boundary_score','z_geo','z_unc'):
        value=aux.get(key)
        if torch.is_tensor(value):arrays[key]=value.detach().float().cpu().numpy()
    np.savez_compressed(out/'evidence.npz',**arrays)
    maps=[(k,v.squeeze()) for k,v in arrays.items() if v.squeeze().ndim==2]
    fig,axes=plt.subplots(1,len(maps),figsize=(4*len(maps),4),squeeze=False)
    for ax,(name,array) in zip(axes[0],maps):
        ax.imshow(array,cmap='magma',vmin=0,vmax=1);ax.set_title(name.replace('_',' '));ax.axis('off')
    fig.tight_layout();fig.savefig(out/'evidence.png',dpi=200);plt.close(fig)


if __name__=='__main__':main()
