"""Measure held-out classification changes after permuting raw clinical variables."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from bua_lel.engine.inference import load_model,predict
from bua_lel.data.preprocessing import read_excel_df
from baselines.metrics import classification_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','image-dir','clinical-table','ids','output-dir'):p.add_argument('--'+key,required=True)
    p.add_argument('--repeats',type=int,default=20);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cpu');a=p.parse_args()
    if a.repeats<1:p.error('--repeats must be positive')
    model,cfg,state=load_model(a.checkpoint,a.device)
    if cfg.modality=='ultrasound_only' or cfg.training_task=='segmentation_only':p.error('Clinical permutation requires a clinical classification model')
    frame=read_excel_df(a.clinical_table);frame=frame.set_index(frame.columns[0])
    ids=[x.strip() for x in Path(a.ids).read_text().splitlines() if x.strip()]
    if not ids or len(ids)!=len(set(ids)):raise ValueError('Identifiers must be unique and nonempty')
    frame=frame.loc[ids];features=frame.iloc[:,:-1].copy();labels=frame.iloc[:,-1].to_numpy(int)
    names=list((state.get('feature_names') or [])[:(state.get('numeric_slice') or (0,0))[1]])+list((state.get('onehot_slices') or {}).keys())
    images={x.stem:x for x in Path(a.image_dir).iterdir() if x.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}}
    def score(table):
        probability=np.stack([predict(model,cfg,state,images[pid],table.loc[pid].to_dict(),a.device)[1] for pid in ids])
        result=classification_metrics(labels,probability)
        return {k:result[k] for k in ('ACC','Macro-F1','Macro-AUC')}
    baseline=score(features);rng=np.random.default_rng(a.seed);rows=[]
    for feature in names:
        for repeat in range(a.repeats):
            altered=features.copy();altered[feature]=rng.permutation(altered[feature].to_numpy())
            scores=score(altered)
            rows.append({'feature':feature,'repeat':repeat,**{k:baseline[k]-scores[k] for k in baseline}})
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'permutation_differences.csv',index=False)
    pd.DataFrame(rows).groupby('feature')[list(baseline)].agg(['mean','std']).to_csv(out/'permutation_summary.csv')
    (out/'baseline.json').write_text(json.dumps(baseline,indent=2),encoding='utf-8')


if __name__=='__main__':main()
