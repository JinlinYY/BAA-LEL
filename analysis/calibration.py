"""Reliability diagrams, expected calibration error, and Brier score."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bua_lel.utils.calibration import BinaryCalibrationAccumulator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',required=True,help='NPZ with probabilities and targets; pixel arrays or N x C class probabilities')
    p.add_argument('--task',choices=['segmentation','classification'],required=True)
    p.add_argument('--bins',type=int,default=15)
    p.add_argument('--output-dir',required=True)
    a=p.parse_args()
    with np.load(a.predictions,allow_pickle=False) as f:
        proba=f['probabilities'] if 'probabilities' in f else f['y_proba']
        target=f['targets'] if 'targets' in f else f['y_true']
    if not np.isfinite(proba).all() or np.any((proba<0)|(proba>1)):raise ValueError('Probabilities must be finite and lie in [0,1]')
    accumulator=BinaryCalibrationAccumulator(a.bins)
    if a.task=='classification':
        target=target.astype(int)
        if proba.ndim!=2 or not np.allclose(proba.sum(1),1,atol=1e-5):raise ValueError('Expected normalized N x C class probabilities')
        accumulator.update(proba.max(1),(proba.argmax(1)==target).astype(float))
        brier=float(np.square(proba-np.eye(proba.shape[1])[target]).sum(1).mean())
    else:
        if not np.isin(target,[0,1]).all():raise ValueError('Segmentation targets must be binary')
        accumulator.update(proba,target);brier=accumulator.compute()['brier_score']
    result=accumulator.compute()
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    metrics={'task':a.task,'ece':result['ece'],'brier_score':brier,'n_observations':result['n_pixels'],'n_bins':a.bins}
    (out/'calibration.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    occupied=result['bin_count']>0
    fig,ax=plt.subplots(figsize=(4,4));ax.plot([0,1],[0,1],'--',color='gray')
    ax.plot(result['bin_confidence'][occupied],result['bin_accuracy'][occupied],'o-',color='#245b8a')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Predicted confidence',ylabel='Observed frequency')
    fig.tight_layout();fig.savefig(out/'reliability.png',dpi=250);plt.close(fig)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
