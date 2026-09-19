"""Evaluate one trained fold on an explicit held-out list of case identifiers."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np,pandas as pd
from baa_lel.engine.inference import load_model,predict
from baa_lel.data.preprocessing import read_excel_df
from baselines.metrics import classification_metrics,segmentation_case_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--image-dir',required=True)
    p.add_argument('--mask-dir',required=True)
    p.add_argument('--clinical-table',required=True)
    p.add_argument('--ids',required=True,help='Text file with one held-out case identifier per line')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--device',default='cpu')
    a=p.parse_args();model,cfg,payload=load_model(a.checkpoint,a.device)
    frame=read_excel_df(a.clinical_table)
    frame=frame.set_index(frame.columns[0])
    ids=Path(a.ids).read_text().splitlines()
    ids=[x.strip() for x in ids if x.strip()]
    if len(set(ids))!=len(ids) or not ids:raise ValueError('Held-out identifiers must be unique and nonempty')
    images={x.stem:x for x in Path(a.image_dir).iterdir() if x.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}}
    masks={x.stem:x for x in Path(a.mask_dir).iterdir() if x.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}}
    rows=[]
    for pid in ids:
        row=frame.loc[pid];prob,cls,_,_=predict(model,cfg,payload,images[pid],row.iloc[:-1].to_dict(),a.device)
        result={'pid':pid}
        if cfg.training_task!='segmentation_only':
            result.update(y_true=int(row.iloc[-1]),y_pred=int(cls.argmax()))
            result.update({f'proba_{i}':float(v) for i,v in enumerate(cls)})
        if cfg.training_task!='classification_only' and pid in masks:
            mask=cv2.imread(str(masks[pid]),cv2.IMREAD_GRAYSCALE)
            if mask is None:raise FileNotFoundError(masks[pid])
            mask=cv2.resize(mask,(cfg.image_size,cfg.image_size),interpolation=cv2.INTER_NEAREST)>0
            result.update(segmentation_case_metrics(prob>=cfg.seg_thr,mask))
        rows.append(result)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    result=pd.DataFrame(rows);result.to_csv(out/'cases.csv',index=False)
    metrics={}
    if cfg.training_task!='segmentation_only':metrics.update(classification_metrics(result.y_true,result[[f'proba_{i}' for i in range(cfg.num_classes)]].values))
    for key in ('Dice','mIoU','HD95','ASSD'):
        if key in result:metrics[key]=float(result[key].mean())
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
