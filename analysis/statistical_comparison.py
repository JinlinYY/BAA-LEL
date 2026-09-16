"""Paired patient-level inference with Holm correction across supplied comparisons."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scipy.stats import binomtest
from baselines.metrics import classification_metrics


def compare(a,b,metric,n=10000,seed=42):
    if a.pid.duplicated().any() or b.pid.duplicated().any() or set(a.pid)!=set(b.pid):
        raise ValueError('Both methods must contain exactly the same unique patients')
    a=a.set_index('pid').sort_index();b=b.set_index('pid').sort_index()
    if 'fold' not in a or 'fold' not in b or not np.array_equal(a.fold,b.fold):
        raise ValueError('Paired comparisons require identical fold assignments')
    rng=np.random.default_rng(seed);size=len(a);indices=np.arange(size)
    classification=metric in {'ACC','Macro-F1','Macro-AUC'}
    if classification:
        if not np.array_equal(a.y_true,b.y_true):raise ValueError('Reference labels differ')
        y=a.y_true.to_numpy(int)
        cols=sorted([c for c in a if c.startswith('proba_')],key=lambda x:int(x.split('_')[-1]))
        if not cols:raise ValueError('Classification requires proba_0, proba_1, ...')
        av=a[cols].values;bv=b[cols].values
        def score(idx,values):return classification_metrics(y[idx],values[idx])[metric]
        groups=[indices[y==c] for c in np.unique(y)]
        def sample():return np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
    else:
        av=a[metric].to_numpy(float);bv=b[metric].to_numpy(float)
        if not np.isfinite(av).all() or not np.isfinite(bv).all():raise ValueError('Paired segmentation metrics must be finite')
        def score(idx,values):return float(values[idx].mean())
        def sample():return rng.choice(indices,size,replace=True)
    delta=score(indices,av)-score(indices,bv);boot=[];extreme=0
    for _ in range(n):
        idx=sample();boot.append(score(idx,av)-score(idx,bv))
        swap=rng.random(size)<0.5
        if classification:swap=swap[:,None]
        difference=score(indices,np.where(swap,bv,av))-score(indices,np.where(swap,av,bv))
        extreme+=abs(difference)>=abs(delta)-1e-12
    probability=(extreme+1)/(n+1)
    test='paired permutation'
    if metric=='ACC':
        ac=av.argmax(1)==y;bc=bv.argmax(1)==y
        n01=int((~ac&bc).sum());n10=int((ac&~bc).sum())
        probability=float(binomtest(n01,n01+n10).pvalue) if n01+n10 else 1.0
        test='exact McNemar'
    low,high=np.percentile(boot,[2.5,97.5])
    return {'metric':metric,'n_pairs':size,'difference':delta,'ci_low':float(low),'ci_high':float(high),'p_value':probability,'test':test}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',required=True);p.add_argument('--baseline',required=True,action='append')
    p.add_argument('--metrics',nargs='+',default=['ACC','Macro-F1','Macro-AUC'])
    p.add_argument('--resamples',type=int,default=10000);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',required=True);a=p.parse_args()
    if a.resamples<1:p.error('--resamples must be positive')
    method=pd.read_csv(a.method,dtype={'pid':str});rows=[]
    for path in a.baseline:
        baseline=pd.read_csv(path,dtype={'pid':str})
        for metric in a.metrics:
            rows.append({'baseline':Path(path).stem,**compare(method,baseline,metric,a.resamples,a.seed)})
    order=np.argsort([r['p_value'] for r in rows]);previous=0.0
    for rank,index in enumerate(order):
        adjusted=min(1.0,max(previous,(len(rows)-rank)*rows[index]['p_value']))
        rows[index]['p_holm']=adjusted;previous=adjusted
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out,index=False)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
