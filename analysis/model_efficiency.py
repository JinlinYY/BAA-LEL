"""Measure parameter count, synchronized latency, and peak accelerator memory."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from bua_lel.engine.inference import load_model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--iterations',type=int,default=30);p.add_argument('--batch-size',type=int,default=1)
    a=p.parse_args()
    if a.iterations<1 or a.warmup<0 or a.batch_size<1:p.error('Invalid iteration or batch count')
    model,cfg,state=load_model(a.checkpoint,a.device)
    x=torch.rand(a.batch_size,3,cfg.image_size,cfg.image_size,device=a.device)
    c=torch.zeros(a.batch_size,state['clinical_dim'],device=a.device)
    cuda=torch.device(a.device).type=='cuda'
    def sync():
        if cuda:torch.cuda.synchronize(a.device)
    durations=[]
    with torch.inference_mode():
        for _ in range(a.warmup):model(x,c_obs=c,m=torch.ones_like(c),task='both')
        sync()
        if cuda:torch.cuda.reset_peak_memory_stats(a.device)
        for _ in range(a.iterations):
            sync();start=time.perf_counter();model(x,c_obs=c,m=torch.ones_like(c),task='both');sync()
            durations.append(1000*(time.perf_counter()-start))
    result={'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'latency_ms_mean':float(np.mean(durations)),'latency_ms_std':float(np.std(durations)),'batch_size':a.batch_size,'iterations':a.iterations,'device':a.device,'precision':'float32','torch':torch.__version__,'input_size':cfg.image_size,'encoder_input_size':cfg.medsam_input_size,'peak_memory_mib':torch.cuda.max_memory_allocated(a.device)/2**20 if cuda else None}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
