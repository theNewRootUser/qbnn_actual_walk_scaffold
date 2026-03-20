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
    slim = []
    for d in out["diagnostics"]:
        slim.append({
            "block_id": d["block_id"],
            "family": d["family"],
            "state_space_size": d["state_space_size"],
            "logical": d.get("logical"),
            "sample_resources": d.get("sample_resources"),
            "transpiled_sample_resources": d.get("transpiled_sample_resources"),
            "transpiled_qpe_resources": d.get("transpiled_qpe_resources"),
            "distribution_metrics": d.get("distribution_metrics"),
        })
    print(json.dumps({"config": out["config"], "diagnostics": slim}, indent=2))

if __name__ == "__main__":
    main()
