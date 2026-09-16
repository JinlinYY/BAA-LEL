"""Train a grid of scale-adaptive morphology radii on shared patient folds."""
import argparse,json,sys
from dataclasses import replace,asdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bua_lel.config import load_config,validate_inputs
from bua_lel.engine.trainer import main as train


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--boundary',nargs='+',type=float,default=[0.1,0.15,0.2])
    p.add_argument('--peritumoral',nargs='+',type=float,default=[0.2,0.3,0.4,0.5])
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();base=load_config(a.config)
    if not a.dry_run:validate_inputs(base)
    for boundary in a.boundary:
        for peritumoral in a.peritumoral:
            if boundary<=0 or peritumoral<=boundary:raise ValueError('Require 0 < boundary < peritumoral')
            name=f'boundary_{boundary:g}_peritumoral_{peritumoral:g}'
            cfg=replace(base,boundary_radius_ratio=boundary,peritumor_radius_ratio=peritumoral,save_dir=str(Path(a.output_dir)/name),experiment_name=name)
            if a.dry_run:print(json.dumps(asdict(cfg)));continue
            if Path(cfg.save_dir).exists():raise FileExistsError(cfg.save_dir)
            train(cfg)


if __name__=='__main__':main()
