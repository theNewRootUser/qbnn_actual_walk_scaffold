# QBNN actual-walk scaffold for LeNet-2 / Zipcode

This scaffold is built around a **blocked hybrid sampler**:

- the full LeNet-2 Bayesian parameter vector lives on the classical side,
- each active local block is discretized and turned into a finite reversible MH chain,
- the quantum side runs an **actual coherent block kernel**:
  - `coherent_mh`: exact controlled row-state preparation of the local MH transition kernel,
  - `szegedy_qpe`: exact Szegedy walk with textbook QPE phase filtering for small local blocks,
- the resulting local samples are embedded back into the full theta vector,
- posterior-predictive scoring is done classically with the same LeNet-2 Bayesian evaluator.

## Important limitations

This is **faithful for small local blocks**. For large local state spaces the code intentionally refuses to build dense walk unitaries and instead exposes local diagnostics so you can catch problems before spending QPU time.

Recommended starting point:

- `discretization_bits = 1` or `2`
- `block_param_count = 1` or `2`
- `coherent_mh` first
- `szegedy_qpe` only after the local diagnostics pass cleanly

## Data

Preferred dataset location:

- `data/zipcode/zipcode.npz`

The NPZ must contain:

- `x_train`, `y_train`, `x_test`, `y_test`

Accepted shapes:

- `x_*`: `[N, 256]` or `[N, 1, 16, 16]`
- `y_*`: `[N]`

Alternative text format:

- `data/zipcode/zip.train`
- `data/zipcode/zip.test`

where each row has `label pixel1 pixel2 ... pixel256`.

## Main commands

```bash
export PYTHONPATH=src

python scripts/run_classical.py --config configs/classical_lenet2_zipcode.yaml > outputs/classical_lenet2_zipcode.json
python scripts/diagnose_local.py --config configs/coherent_mh_qpe_ideal_lenet2.yaml > outputs/diag_coherent_mh.json
python scripts/diagnose_local.py --config configs/szegedy_qpe_ideal_lenet2.yaml > outputs/diag_szegedy_qpe.json

python scripts/run_quantum.py --config configs/coherent_mh_qpe_ideal_lenet2.yaml > outputs/coherent_mh_qpe_ideal_lenet2.json
python scripts/run_quantum.py --config configs/coherent_mh_qpe_noisy_lenet2.yaml > outputs/coherent_mh_qpe_noisy_lenet2.json
python scripts/run_quantum.py --config configs/coherent_mh_qpe_ibm_lenet2.yaml > outputs/coherent_mh_qpe_ibm_lenet2.json

python scripts/run_quantum.py --config configs/szegedy_qpe_ideal_lenet2.yaml > outputs/szegedy_qpe_ideal_lenet2.json
python scripts/run_quantum.py --config configs/szegedy_qpe_noisy_lenet2.yaml > outputs/szegedy_qpe_noisy_lenet2.json
python scripts/run_quantum.py --config configs/szegedy_qpe_ibm_lenet2.yaml > outputs/szegedy_qpe_ibm_lenet2.json

python scripts/transpile_report.py --config configs/coherent_mh_qpe_ideal_lenet2.yaml
python scripts/transpile_report.py --config configs/szegedy_qpe_ideal_lenet2.yaml
```

## Output shape

Quantum runs save:

- local/block diagnostics
- circuit resource reports
- per-sweep full-theta posterior samples
- posterior-predictive metrics from the classical LeNet-2 evaluator
