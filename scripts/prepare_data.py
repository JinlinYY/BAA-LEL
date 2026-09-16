"""Convert dataset layouts to matched image/mask files and a clinical CSV."""
import argparse
from pathlib import Path
import shutil,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import pandas as pd
from baselines.protocol import protocol_records
from baselines.data import load_clinical_frame


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,choices=['HER2USC','LMNUSC','BrEaST','BUSI'])
    p.add_argument('--source-root',required=True,help='Parent of HER2, LNM, BrEaST, or BUSI')
    p.add_argument('--output-root',default='data')
    args=p.parse_args()
    records=protocol_records(args.dataset,args.source_root)
    clinical=load_clinical_frame(args.dataset,Path(args.source_root),records)
    out=Path(args.output_root)/args.dataset
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    for folder in ('images','masks'):(out/folder).mkdir(parents=True,exist_ok=True)
    rows=[]
    for index,record in enumerate(records):
        pid=f'case_{index:06d}'
        shutil.copyfile(record.image_path,out/'images'/f'{pid}{record.image_path.suffix.lower()}')
        if record.mask_paths:
            image=cv2.imread(str(record.image_path),cv2.IMREAD_GRAYSCALE)
            if image is None:raise FileNotFoundError(record.image_path)
            mask=np.zeros(image.shape,np.uint8)
            for path in record.mask_paths:
                part=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
                if part is None:raise FileNotFoundError(path)
                if part.shape!=mask.shape:part=cv2.resize(part,(mask.shape[1],mask.shape[0]),interpolation=cv2.INTER_NEAREST)
                mask=np.maximum(mask,(part>0).astype(np.uint8)*255)
            cv2.imwrite(str(out/'masks'/f'{pid}.png'),mask)
        row={'patient_id':pid,**clinical.loc[record.pid].to_dict(),'label':record.label}
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'clinical.csv',index=False)
    pd.DataFrame({'patient_id':[r['patient_id'] for r in rows],'source_id':[r.pid for r in records]}).to_csv(out/'source_mapping.csv',index=False)
    print(f'Prepared {len(rows)} cases under {out}. Keep source_mapping.csv private.')


if __name__=='__main__':main()
