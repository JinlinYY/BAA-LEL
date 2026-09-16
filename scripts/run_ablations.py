"""Execute controlled ablations with the same patient folds."""
import argparse,subprocess,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',action='append',help='Repeat to select YAML files; default: all configs/ablations/*.yaml')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--device')
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    configs=a.config or [str(x) for x in sorted((root/'configs/ablations').glob('*.yaml'))]
    for config in configs:
        cmd=[sys.executable,str(root/'scripts/cross_validate.py'),'--config',config]
        if a.dry_run:cmd+=['--dry-run']
        if a.device:cmd+=['--device',a.device]
        subprocess.run(cmd,check=True)


if __name__=='__main__':main()
