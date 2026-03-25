from __future__ import annotations

import argparse
import json
from pathlib import Path
import os, sys
import numpy as np
PATH = r"C:\Users\dario\Downloads\qbnn_actual_walk_scaffold\qbnn_actual_walk_scaffold"
sys.path.append(PATH)
from src.qbnn.config import load_config
from src.qbnn.models import build_bayesian_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to experiment yaml")
    ap.add_argument("--out", required=True, help="Output json path")
    ap.add_argument(
        "--scale",
        type=float,
        default=0.05,
        help="Relative scale vs prior_std. 0.05 = small prior-like init, 1.0 = exact prior scale.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="RNG seed for reproducible initialization.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = build_bayesian_model(cfg.model)

    sigma = float(cfg.model.prior_std) * float(args.scale)
    rng = np.random.default_rng(args.seed)
    theta = rng.normal(loc=0.0, scale=sigma, size=model.num_params).astype(np.float64)

    out = {
        "result": {
            "theta_map": theta.tolist()
        }
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f)

    print(
        json.dumps(
            {
                "out": str(out_path),
                "num_params": int(model.num_params),
                "prior_std": float(cfg.model.prior_std),
                "scale": float(args.scale),
                "actual_sigma": float(sigma),
                "seed": int(args.seed),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()