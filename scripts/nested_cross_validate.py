"""Run joint models with inner validation and outer patient-level evaluation."""
import argparse,json,sys
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from baselines.multitask_runner import MultitaskConfig,run_multitask_benchmark


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--device');p.add_argument('--data-root');p.add_argument('--output-root')
    a=p.parse_args();values=yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    for key in ('device','data_root','output_root'):
        if getattr(a,key):values[key]=getattr(a,key)
    cfg=MultitaskConfig(**values)
    if a.dry_run:print(json.dumps(asdict(cfg),indent=2,ensure_ascii=False));return
    print(json.dumps(run_multitask_benchmark(cfg),indent=2,default=float))


if __name__=='__main__':main()
