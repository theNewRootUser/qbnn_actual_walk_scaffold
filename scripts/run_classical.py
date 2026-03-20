from __future__ import annotations
import argparse, json
import os, sys
sys.path.append('/home/dario/Desktop/qbnn_actual_walk_scaffold')
from src.qbnn.config import load_config
from src.qbnn.experiments import run_classical_baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = run_classical_baseline(cfg)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
