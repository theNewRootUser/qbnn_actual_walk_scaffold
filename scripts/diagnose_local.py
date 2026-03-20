from __future__ import annotations
import argparse, json

import os, sys
sys.path.append('/home/dario/Desktop/qbnn_actual_walk_scaffold')
from src.qbnn.config import load_config
from src.qbnn.experiments import diagnose_quantum_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = diagnose_quantum_experiment(cfg)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
