"""Predict lesion probability and class probabilities for one image."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from baa_lel.engine.inference import load_model,predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--image',required=True)
    p.add_argument('--clinical-json',help='JSON object containing raw clinical variables; no label is required')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    model,cfg,payload=load_model(args.checkpoint,args.device)
    values=json.loads(Path(args.clinical_json).read_text()) if args.clinical_json else {}
    if payload['clinical_dim'] and not args.clinical_json:
        p.error('--clinical-json is required for a multimodal checkpoint')
    probability,classes,_,shape=predict(model,cfg,payload,args.image,values,args.device)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    np.save(out/'lesion_probability.npy',probability)
    mask=cv2.resize((probability>=cfg.seg_thr).astype('uint8'),(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)*255
    cv2.imwrite(str(out/'lesion_mask.png'),mask)
    result={'image':Path(args.image).name,'threshold':cfg.seg_thr}
    if cfg.training_task!='segmentation_only':
        result.update(predicted_class=int(classes.argmax()),class_probabilities=classes.tolist())
    (out/'prediction.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
