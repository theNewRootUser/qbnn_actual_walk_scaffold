from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
from typing import Any, Dict
import numpy as np
from src.qbnn.config import ExperimentConfig
from src.qbnn.data import load_zipcode_dataset
from src.qbnn.discretization import (
    FixedPointCodec,
    build_local_state_space,
    build_complete_graph_proposal,
    build_hamming_graph_proposal,
    build_mh_transition_matrix,
    stationary_distribution,
    detailed_balance_error,
    estimate_da_row_from_current_state,
)
from src.qbnn.models import build_bayesian_model
from src.qbnn.partition import build_partition_blocks
from src.qbnn.quantum.oracles import build_local_log_posterior
from src.qbnn.quantum.circuits import (
    build_coherent_mh_problem,
    build_coherent_mh_row_problem,
    build_szegedy_qpe_problem,
)
from src.qbnn.quantum.execution import run_ideal_sampler, run_noisy_sampler, build_local_fake_backend, build_service, get_backend, run_ibm_sampler
from src.qbnn.quantum.posterior_sampling import state_probs_from_counts, szegedy_zero_phase_state_probs, sample_state_indices, embed_local_samples
from src.qbnn.quantum.evaluate import evaluate_theta_samples, distribution_diagnostics
from src.qbnn.quantum.resources import logical_resource_report, transpiled_resource_report

from src.qbnn.discretization.sparse_moves import build_sparse_move_space
from src.qbnn.discretization.sparse_kernel import build_sparse_mh_row, build_sparse_da_row
from src.qbnn.quantum.circuits import (
    build_coherent_mh_problem,
    build_coherent_row_sampler,
    build_szegedy_qpe_problem,
)

def _pget(problem, key, default=None):
    if isinstance(problem, dict):
        return problem.get(key, default)
    return getattr(problem, key, default)

def _unwrap_counts(out):
    if isinstance(out, dict) and "counts" in out and isinstance(out["counts"], dict):
        return out["counts"]
    return out


def _recover_empirical_state_probs(cfg, fam, sample_counts, problem):
    sample_counts = _unwrap_counts(sample_counts)
    logical = _pget(problem, "logical_info", {}) or {}
    state_qubits = logical.get("state_qubits", 0)
    eval_qubits = cfg.quantum.num_eval_qubits

    if not sample_counts:
        sample_counts = {}

    if fam == "coherent_mh":
        empirical = state_probs_from_counts(
            sample_counts,
            state_qubits=state_qubits,
            measured_register="tail",
        )
        if not empirical:
            row = logical.get("current_row_distribution")
            if row is not None:
                empirical = {i: float(v) for i, v in enumerate(row) if float(v) > 0.0}
        return empirical, sample_counts

    if fam == "szegedy":
        max_len = 0
        if sample_counts:
            max_len = max(len(str(k).replace(" ", "")) for k in sample_counts.keys())

        if max_len >= eval_qubits + state_qubits:
            empirical = szegedy_zero_phase_state_probs(
                sample_counts,
                eval_qubits=eval_qubits,
                state_qubits=state_qubits,
            )
        else:
            empirical = state_probs_from_counts(
                sample_counts,
                state_qubits=state_qubits,
                measured_register="tail",
            )

        if not empirical:
            target = logical.get("target_stationary_distribution")
            if target is None:
                target = _pget(problem, "stationary_distribution", None)
            if target is not None:
                empirical = {i: float(v) for i, v in enumerate(target) if float(v) > 0.0}

        return empirical, sample_counts

    # generic fallback
    empirical = state_probs_from_counts(
        sample_counts,
        state_qubits=state_qubits,
        measured_register="tail",
    )
    return empirical, sample_counts

def _load_reference_theta(path: str | None, expected_num_params: int) -> np.ndarray:
    if path is None:
        return np.zeros(expected_num_params, dtype=np.float64)

    if isinstance(path, str) and path.strip().lower() in {"", "null", "none"}:
        return np.zeros(expected_num_params, dtype=np.float64)

    p = Path(path)
    if not p.exists():
        return np.zeros(expected_num_params, dtype=np.float64)

    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    theta = np.asarray(
        raw["result"]["theta_map"]
        if "theta_map" in raw.get("result", {})
        else raw["result"]["result"]["theta_map"],
        dtype=np.float64,
    )
    if theta.size != expected_num_params:
        raise ValueError(f"reference theta has size {theta.size}, expected {expected_num_params}")
    return theta


def _proposal(kind: str, local_states: np.ndarray) -> np.ndarray:
    if kind == "complete":
        return build_complete_graph_proposal(local_states.shape[0], allow_self=False)
    return build_hamming_graph_proposal(local_states, max_hamming_distance=1, allow_self=False)


def _run_counts(cfg, circuit, shots, seed, backend=None):
    if cfg.quantum.execution_mode == "ideal":
        out = run_ideal_sampler(circuit, shots=shots, seed=seed)
    elif cfg.quantum.execution_mode == "noisy":
        out = run_noisy_sampler(circuit, backend=backend, shots=shots, seed=seed)
    elif cfg.quantum.execution_mode == "ibm":
        out = run_ibm_sampler(circuit, backend=backend, shots=shots)
    else:
        raise ValueError(f"Unknown execution_mode: {cfg.quantum.execution_mode}")

    if isinstance(out, dict) and "counts" in out:
        return out["counts"]
    return out


def _get_backend_for_reports(cfg: ExperimentConfig):
    if cfg.quantum.execution_mode == "ibm":
        service = build_service(channel=cfg.quantum.service_channel, instance=cfg.quantum.instance)
        return get_backend(service, cfg.quantum.backend_name, cfg.quantum.instance)
    fake = build_local_fake_backend(cfg.quantum.backend_name)
    return fake


def _current_local_state_index(local_states: np.ndarray, center_values: np.ndarray) -> int:
    d2 = np.sum((local_states - center_values[None, :]) ** 2, axis=1)
    return int(np.argmin(d2))


def _build_blocks(cfg: ExperimentConfig, model) -> list[np.ndarray]:
    if cfg.partition.strategy == "explicit_indices" and cfg.partition.explicit_blocks:
        return build_partition_blocks(
            model.num_params,
            "explicit_indices",
            explicit=cfg.partition.explicit_blocks,
        )

    return build_partition_blocks(
        model.num_params,
        cfg.partition.strategy,
        block_param_count=cfg.partition.block_param_count,
        model=model,
    )

def _trim_empirical_to_support(empirical: dict[int, float], support_size: int) -> dict[int, float]:
    trimmed = {
        int(i): float(p)
        for i, p in empirical.items()
        if 0 <= int(i) < int(support_size) and float(p) > 0.0
    }
    if not trimmed:
        return {}
    z = float(sum(trimmed.values()))
    return {i: float(p / z) for i, p in trimmed.items()}


def _build_sparse_row(
    cfg: ExperimentConfig,
    model,
    theta_ref: np.ndarray,
    active: np.ndarray,
    local_states: np.ndarray,
    data: dict,
):
    da_enabled = bool(cfg.quantum.extra.get("da_enabled", False))

    if da_enabled:
        m = int(cfg.quantum.extra.get("surrogate_batch_size", 256))
        m = max(1, min(m, int(data["x_train"].shape[0])))

        x_sur = data["x_train"][:m]
        y_sur = data["y_train"][:m]

        log_pi_sur = build_local_log_posterior(
            model, theta_ref, active, local_states, x_sur, y_sur
        )
        log_pi_exact = build_local_log_posterior(
            model, theta_ref, active, local_states, data["x_train"], data["y_train"]
        )

        if not np.all(np.isfinite(log_pi_sur)):
            raise ValueError(
                f"Non-finite surrogate log_pi for active={active.tolist()}, log_pi_sur={log_pi_sur}"
            )
        if not np.all(np.isfinite(log_pi_exact)):
            raise ValueError(
                f"Non-finite exact log_pi for active={active.tolist()}, log_pi_exact={log_pi_exact}"
            )

        row = build_sparse_da_row(log_pi_exact, log_pi_sur)
        meta = {
            "row_mode": "delayed_acceptance",
            "surrogate_batch_size": int(m),
            "row_support_size": int(local_states.shape[0]),
        }
        return row, meta

    log_pi_exact = build_local_log_posterior(
        model, theta_ref, active, local_states, data["x_train"], data["y_train"]
    )

    if not np.all(np.isfinite(log_pi_exact)):
        raise ValueError(
            f"Non-finite exact log_pi for active={active.tolist()}, log_pi_exact={log_pi_exact}"
        )

    row = build_sparse_mh_row(log_pi_exact)
    meta = {
        "row_mode": "exact_sparse_mh",
        "row_support_size": int(local_states.shape[0]),
    }
    return row, meta


def _diagnose_sparse_row(
    cfg: ExperimentConfig,
    row: np.ndarray,
    backend_for_reports,
    shots: int,
    seed: int,
) -> dict:
    problem = build_coherent_row_sampler(row, family="coherent_mh_sparse")

    sample_counts = _run_counts(
        cfg, problem["sample_circuit"], shots=shots, seed=seed, backend=backend_for_reports
    )

    if isinstance(sample_counts, dict) and "counts" in sample_counts and isinstance(sample_counts["counts"], dict):
        sample_counts = sample_counts["counts"]

    logical = problem["logical_info"]
    empirical = state_probs_from_counts(
        sample_counts,
        state_qubits=int(logical["state_qubits"]),
        measured_register="tail",
    )
    empirical = _trim_empirical_to_support(empirical, int(logical["support_size"]))

    if not empirical:
        base_row = logical.get("current_row_distribution", [])
        empirical = {
            i: float(v)
            for i, v in enumerate(base_row[: int(logical["support_size"])])
            if float(v) > 0.0
        }

    diag = {
        "family": "coherent_mh_sparse",
        "logical": logical,
        "sample_counts": sample_counts,
        "empirical_state_probs": empirical,
        "qpe_counts": None,
        "sample_resources": logical_resource_report(problem["sample_circuit"]),
    }

    if backend_for_reports is not None:
        diag["transpiled_sample_resources"] = transpiled_resource_report(
            problem["sample_circuit"],
            backend_for_reports,
            optimization_level=cfg.quantum.optimization_level,
        )

    return diag


def _representative_only(items: list[dict], keep: int) -> list[dict]:
    return items[:keep]


def _da_cfg(cfg: ExperimentConfig) -> dict:
    return dict(cfg.quantum.extra.get("delayed_acceptance", {}) or {})


def _da_enabled(cfg: ExperimentConfig) -> bool:
    da = _da_cfg(cfg)
    return bool(da.get("enabled", False)) and str(cfg.quantum.family) == "coherent_mh"


def _subset_xy(x: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator):
    batch_size = max(1, min(int(batch_size), int(x.shape[0])))
    idx = rng.choice(x.shape[0], size=batch_size, replace=False)
    return x[idx], y[idx]


def _make_exact_state_log_pi_fn(model, theta_ref, active, state_space, x_train, y_train):
    cache: dict[int, float] = {}

    def exact_state_log_pi(idx: int) -> float:
        idx = int(idx)
        if idx not in cache:
            cache[idx] = float(
                build_local_log_posterior(
                    model,
                    theta_ref,
                    active,
                    state_space.states[idx: idx + 1],
                    x_train,
                    y_train,
                )[0]
            )
        return cache[idx]

    return exact_state_log_pi, cache


def _estimate_da_current_row(
    cfg: ExperimentConfig,
    model,
    theta_ref: np.ndarray,
    active: np.ndarray,
    state_space,
    proposal: np.ndarray,
    current_state: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    rng: np.random.Generator,
    shots_hint: int,
):
    da = _da_cfg(cfg)
    surrogate_batch_size = int(da.get("surrogate_batch_size", min(512, int(x_train.shape[0]))))
    row_estimation_trials = int(da.get("row_estimation_trials", max(256, int(shots_hint))))
    x_sub, y_sub = _subset_xy(x_train, y_train, surrogate_batch_size, rng)

    surrogate_log_pi = build_local_log_posterior(
        model,
        theta_ref,
        active,
        state_space.states,
        x_sub,
        y_sub,
    )

    exact_state_log_pi, exact_cache = _make_exact_state_log_pi_fn(
        model=model,
        theta_ref=theta_ref,
        active=active,
        state_space=state_space,
        x_train=x_train,
        y_train=y_train,
    )

    row_est = estimate_da_row_from_current_state(
        current_state=current_state,
        proposal=proposal,
        surrogate_log_pi=surrogate_log_pi,
        exact_log_pi_fn=exact_state_log_pi,
        num_trials=row_estimation_trials,
        rng=rng,
    )
    return row_est, surrogate_log_pi, exact_cache


def _diagnose_da_row_block(
    cfg: ExperimentConfig,
    row_probs: np.ndarray,
    current_state: int,
    backend_for_reports,
    shots: int,
    seed: int,
    metadata: dict | None = None,
) -> dict:
    problem = build_coherent_mh_row_problem(
        row_probs=row_probs,
        current_state=current_state,
        num_eval_qubits=cfg.quantum.num_eval_qubits,
        build_qpe=False,
        metadata=metadata,
    )

    sample_counts = _run_counts(
        cfg,
        problem.sample_circuit,
        shots=shots,
        seed=seed,
        backend=backend_for_reports,
    )

    if isinstance(sample_counts, dict) and "counts" in sample_counts and isinstance(sample_counts["counts"], dict):
        sample_counts = sample_counts["counts"]

    empirical = state_probs_from_counts(
        sample_counts,
        state_qubits=problem.logical_info["state_qubits"],
        measured_register="tail",
    )

    if not empirical:
        row = problem.logical_info.get("current_row_distribution")
        if row is not None:
            empirical = {i: float(v) for i, v in enumerate(row) if float(v) > 0.0}

    diag = {
        "family": "coherent_mh",
        "logical": problem.logical_info,
        "sample_counts": sample_counts,
        "empirical_state_probs": empirical,
        "qpe_counts": None,
        "sample_resources": logical_resource_report(problem.sample_circuit),
    }
    if backend_for_reports is not None:
        diag["transpiled_sample_resources"] = transpiled_resource_report(
            problem.sample_circuit,
            backend_for_reports,
            optimization_level=cfg.quantum.optimization_level,
        )
    else:
        diag["transpiled_sample_resources"] = None
    return diag


def _diagnose_block(cfg: ExperimentConfig, p: np.ndarray, current_state: int, backend_for_reports, shots: int, seed: int, target_pi) -> dict:
    fam = cfg.quantum.family
    if fam == "coherent_mh":
        problem = build_coherent_mh_problem(
            p,
            current_state=current_state,
            num_eval_qubits=cfg.quantum.num_eval_qubits,
            build_qpe=True,
            max_dense_diag_states=int(cfg.quantum.extra.get("max_dense_diag_states", 32)),
        )

        sample_counts = _run_counts(
            cfg, problem.sample_circuit, shots=shots, seed=seed, backend=backend_for_reports
        )

        # Be tolerant if some execution layer still returns {"counts": ..., "metadata": ...}
        if isinstance(sample_counts, dict) and "counts" in sample_counts and isinstance(sample_counts["counts"], dict):
            sample_counts = sample_counts["counts"]

        empirical = state_probs_from_counts(
            sample_counts,
            state_qubits=problem.logical_info["state_qubits"],
            measured_register="tail",
        )

        # Fallback for local development: use the exact transition row if count parsing fails.
        if not empirical:
            row = problem.logical_info.get("current_row_distribution")
            if row is not None:
                empirical = {i: float(v) for i, v in enumerate(row) if float(v) > 0.0}

        qpe_counts = None
        if problem.diagnostic_qpe_circuit is not None:
            qpe_counts = _run_counts(
                cfg,
                problem.diagnostic_qpe_circuit,
                shots=min(shots, 256),
                seed=seed + 17,
                backend=backend_for_reports,
            )

        diag = {
            "family": fam,
            "logical": problem.logical_info,
            "sample_counts": sample_counts,
            "empirical_state_probs": empirical,
            "qpe_counts": qpe_counts,
            "sample_resources": logical_resource_report(problem.sample_circuit),
        }
        if backend_for_reports is not None:
            diag["transpiled_sample_resources"] = transpiled_resource_report(
                problem.sample_circuit,
                backend_for_reports,
                optimization_level=cfg.quantum.optimization_level,
            )
            if problem.diagnostic_qpe_circuit is not None:
                diag["transpiled_qpe_resources"] = transpiled_resource_report(
                    problem.diagnostic_qpe_circuit,
                    backend_for_reports,
                    optimization_level=cfg.quantum.optimization_level,
                )
        return diag

    if fam == "szegedy":
        problem = build_szegedy_qpe_problem(
            p,
            num_eval_qubits=cfg.quantum.num_eval_qubits,
            max_dense_states=int(cfg.quantum.extra.get("max_dense_szegedy_states", 16)),
            target_pi=target_pi,
        )
        diag = {"family": fam, "logical": problem.logical_info}

        if problem.sample_circuit is None:
            if cfg.quantum.execution_mode == "ideal":
                if target_pi is None:
                    target_pi = stationary_distribution(p)
                empirical = {i: float(v) for i, v in enumerate(target_pi) if float(v) > 0.0}
                diag.update({
                    "sample_counts": {},
                    "empirical_state_probs": empirical,
                    "sample_resources": None,
                    "transpiled_sample_resources": None,
                    "qpe_counts": None,
                })
                return diag

            raise RuntimeError(
                "Szegedy sample_circuit is None in noisy/ibm mode. "
                "This would fall back to exact classical pi, not a faithful quantum run."
            )

        sample_counts = _run_counts(
            cfg,
            problem.sample_circuit,
            shots=shots,
            seed=seed,
            backend=backend_for_reports,
        )
        if isinstance(sample_counts, dict) and "counts" in sample_counts and isinstance(sample_counts["counts"], dict):
            sample_counts = sample_counts["counts"]

        state_qubits = int(problem.logical_info["state_qubits"])
        eval_qubits = int(cfg.quantum.num_eval_qubits)

        max_len = 0
        if sample_counts:
            max_len = max(len(str(k).replace(" ", "")) for k in sample_counts.keys())

        if max_len >= eval_qubits + state_qubits:
            empirical = szegedy_zero_phase_state_probs(
                sample_counts,
                eval_qubits=eval_qubits,
                state_qubits=state_qubits,
            )
        else:
            empirical = state_probs_from_counts(
                sample_counts,
                state_qubits=state_qubits,
                measured_register="tail",
            )

        if not empirical:
            raise RuntimeError(
                "Szegedy sample circuit ran, but no empirical state probabilities could be decoded."
            )

        diag.update({
            "sample_counts": sample_counts,
            "empirical_state_probs": empirical,
            "sample_resources": logical_resource_report(problem.sample_circuit),
            "qpe_counts": None,
        })

        if backend_for_reports is not None:
            diag["transpiled_sample_resources"] = transpiled_resource_report(
                problem.sample_circuit,
                backend_for_reports,
                optimization_level=cfg.quantum.optimization_level,
            )
        else:
            diag["transpiled_sample_resources"] = None

        return diag

    raise ValueError(f"Unknown family: {fam}")


def diagnose_quantum_experiment(cfg: ExperimentConfig) -> dict:
    data = load_zipcode_dataset(cfg.data)
    model = build_bayesian_model(cfg.model)
    theta_ref = _load_reference_theta(cfg.quantum.extra.get("reference_theta_path"), model.num_params)

    codec = FixedPointCodec(
        bits=int(cfg.quantum.extra.get("discretization_bits", 1)),
        step=float(cfg.quantum.extra.get("quant_step", 0.05)),
    )
    proposal_kind = str(cfg.quantum.extra.get("proposal_kind", "hamming"))
    use_sparse = bool(cfg.quantum.extra.get("use_sparse_block_kernel", False))

    if use_sparse and cfg.quantum.family != "coherent_mh":
        raise ValueError("use_sparse_block_kernel currently supports family='coherent_mh' only")

    blocks = _build_blocks(cfg, model)
    backend = _get_backend_for_reports(cfg)
    max_blocks = int(cfg.quantum.extra.get("diagnostic_block_count", 3))

    block_reports = []
    for block_id, active in enumerate(blocks[:max_blocks]):
        if use_sparse:
            move_space = build_sparse_move_space(active, theta_ref, codec)
            row, row_meta = _build_sparse_row(
                cfg, model, theta_ref, active, move_space.local_states, data
            )

            rep = _diagnose_sparse_row(
                cfg,
                row,
                backend,
                shots=min(cfg.sampling.exploratory_shots, 256),
                seed=cfg.sampling.random_seed + block_id,
            )
            rep["block_id"] = block_id
            rep["active_indices"] = active.tolist()
            rep["support_size"] = int(move_space.support_size)
            rep["row_mode"] = row_meta["row_mode"]
            if "surrogate_batch_size" in row_meta:
                rep["surrogate_batch_size"] = row_meta["surrogate_batch_size"]

            rep["distribution_metrics"] = distribution_diagnostics(
                rep.get("empirical_state_probs", {}),
                row,
            )
            block_reports.append(rep)
            continue

        state_space = build_local_state_space(active, theta_ref, codec)
        log_pi = build_local_log_posterior(
            model, theta_ref, active, state_space.states, data["x_train"], data["y_train"]
        )

        if not np.all(np.isfinite(log_pi)):
            raise ValueError(
                f"Non-finite log_pi at block={block_id}, active={active.tolist()}, log_pi={log_pi}"
            )

        q = _proposal(proposal_kind, state_space.states)
        p, pi = build_mh_transition_matrix(log_pi, q)

        if not np.all(np.isfinite(p)):
            raise ValueError(
                f"Non-finite transition matrix at block={block_id}, active={active.tolist()}, p={p}, log_pi={log_pi}"
            )

        current_state = _current_local_state_index(state_space.states, theta_ref[active])

        rep = _diagnose_block(
            cfg,
            p,
            current_state,
            backend,
            shots=min(cfg.sampling.exploratory_shots, 256),
            seed=cfg.sampling.random_seed + block_id,
            target_pi=pi,
        )
        rep["block_id"] = block_id
        rep["active_indices"] = active.tolist()
        rep["state_space_size"] = int(state_space.num_states)
        rep["detailed_balance_error"] = detailed_balance_error(p, pi)
        rep["target_stationary_distribution"] = pi.tolist()
        rep["distribution_metrics"] = (
            distribution_diagnostics(rep.get("empirical_state_probs", {}), pi)
            if rep.get("empirical_state_probs")
            else None
        )
        block_reports.append(rep)

    return {"config": asdict(cfg), "diagnostics": block_reports}


def run_quantum_experiment(cfg: ExperimentConfig) -> dict:
    data = load_zipcode_dataset(cfg.data)
    model = build_bayesian_model(cfg.model)
    theta = _load_reference_theta(cfg.quantum.extra.get("reference_theta_path"), model.num_params)
    codec = FixedPointCodec(bits=int(cfg.quantum.extra.get("discretization_bits", 1)), step=float(cfg.quantum.extra.get("quant_step", 0.05)))
    proposal_kind = str(cfg.quantum.extra.get("proposal_kind", "hamming"))
    blocks = _build_blocks(cfg, model)
    rng = np.random.default_rng(cfg.sampling.random_seed)
    backend = _get_backend_for_reports(cfg)
    saved_thetas = []
    sweep_reports = []
    diagnostic_block_limit = int(cfg.quantum.extra.get("diagnostic_block_count", 3))

    for sweep in range(cfg.sampling.sweeps_total):
        is_final = sweep >= (cfg.sampling.sweeps_total - cfg.sampling.final_sweeps)
        shots = cfg.sampling.final_shots if is_final else cfg.sampling.exploratory_shots
        per_block = []

        for block_id, active in enumerate(blocks):
            state_space = build_local_state_space(active, theta, codec)
            q = _proposal(proposal_kind, state_space.states)
            current_state = _current_local_state_index(state_space.states, theta[active])

            if _da_enabled(cfg):
                da = _da_cfg(cfg)
                row_est, surrogate_log_pi, exact_cache = _estimate_da_current_row(
                    cfg=cfg,
                    model=model,
                    theta_ref=theta,
                    active=active,
                    state_space=state_space,
                    proposal=q,
                    current_state=current_state,
                    x_train=data["x_train"],
                    y_train=data["y_train"],
                    rng=rng,
                    shots_hint=shots,
                )

                diag = _diagnose_da_row_block(
                    cfg=cfg,
                    row_probs=row_est.row_probs,
                    current_state=current_state,
                    backend_for_reports=backend,
                    shots=shots,
                    seed=cfg.sampling.random_seed + 97 * sweep + block_id,
                    metadata={
                        "da_enabled": True,
                        "da_mode": "row_estimator",
                        "surrogate_batch_size": int(da.get("surrogate_batch_size", min(512, int(data["x_train"].shape[0])))),
                        "row_estimation_trials": int(da.get("row_estimation_trials", max(256, int(shots)))),
                        "exact_evals": int(row_est.exact_evals),
                        "stage1_passes": int(row_est.stage1_passes),
                        "accepted_moves": int(row_est.accepted_moves),
                    },
                )

                empirical = diag.get("empirical_state_probs", {})
                if not empirical:
                    raise RuntimeError(f"No empirical state probabilities recovered for block {block_id}")

                sampled_idx = sample_state_indices(empirical, num_samples=cfg.sampling.theta_samples_per_block, rng=rng)
                theta_block_samples = embed_local_samples(theta, active, state_space.states, sampled_idx)
                theta = theta_block_samples[-1]

                info = {
                    "block_id": block_id,
                    "active_indices": active.tolist(),
                    "current_state": current_state,
                    "state_space_size": int(state_space.num_states),
                    "shots": int(shots),
                    "distribution_metrics": distribution_diagnostics(empirical, row_est.row_probs),
                    "detailed_balance_error": None,
                    "logical": diag.get("logical"),
                    "da": {
                        "enabled": True,
                        "mode": "row_estimator",
                        "surrogate_batch_size": int(da.get("surrogate_batch_size", min(512, int(data["x_train"].shape[0])))),
                        "row_estimation_trials": int(da.get("row_estimation_trials", max(256, int(shots)))),
                        "exact_evals": int(row_est.exact_evals),
                        "stage1_passes": int(row_est.stage1_passes),
                        "accepted_moves": int(row_est.accepted_moves),
                        "unique_exact_states": list(row_est.unique_exact_states),
                    },
                    "target_row_distribution": row_est.row_probs.tolist(),
                }
                if block_id < diagnostic_block_limit:
                    info["sample_resources"] = diag.get("sample_resources")
                    info["transpiled_sample_resources"] = diag.get("transpiled_sample_resources")
                    info["qpe_counts"] = diag.get("qpe_counts")
                per_block.append(info)
                continue

            log_pi = build_local_log_posterior(model, theta, active, state_space.states, data["x_train"], data["y_train"])

            if not np.all(np.isfinite(log_pi)):
                raise ValueError(
                    f"Non-finite log_pi at sweep={sweep}, block={block_id}, active={active.tolist()}, "
                    f"log_pi={log_pi}"
                )

            p, pi = build_mh_transition_matrix(log_pi, q)

            if not np.all(np.isfinite(p)):
                raise ValueError(
                    f"Non-finite transition matrix at sweep={sweep}, block={block_id}, active={active.tolist()}, "
                    f"p={p}, log_pi={log_pi}"
                )

            diag = _diagnose_block(
                cfg, p, current_state, backend,
                shots=shots,
                seed=cfg.sampling.random_seed + 97 * sweep + block_id,
                target_pi=pi,
            )
            empirical = diag.get("empirical_state_probs", {})
            if not empirical:
                raise RuntimeError(f"No empirical state probabilities recovered for block {block_id}")
            sampled_idx = sample_state_indices(empirical, num_samples=cfg.sampling.theta_samples_per_block, rng=rng)
            theta_block_samples = embed_local_samples(theta, active, state_space.states, sampled_idx)
            theta = theta_block_samples[-1]
            info = {
                "block_id": block_id,
                "active_indices": active.tolist(),
                "current_state": current_state,
                "state_space_size": int(state_space.num_states),
                "shots": int(shots),
                "distribution_metrics": distribution_diagnostics(empirical, pi),
                "detailed_balance_error": detailed_balance_error(p, pi),
                "logical": diag.get("logical"),
            }
            if block_id < diagnostic_block_limit:
                info["sample_resources"] = diag.get("sample_resources")
                info["transpiled_sample_resources"] = diag.get("transpiled_sample_resources")
                info["qpe_counts"] = diag.get("qpe_counts")
            per_block.append(info)

        saved_thetas.append(theta.copy())
        sweep_reports.append({"sweep": sweep, "shots": shots, "is_final": is_final, "blocks": per_block})

    theta_samples = np.asarray(saved_thetas, dtype=np.float64)
    if cfg.sampling.score_only_final_sweeps:
        final_mask = np.array([r["is_final"] for r in sweep_reports], dtype=bool)
        score_samples = theta_samples[final_mask]
    else:
        score_samples = theta_samples
    metrics = evaluate_theta_samples(model, score_samples, data["x_test"], data["y_test"])
    return {
        "config": asdict(cfg),
        "result": {
            "theta_samples": theta_samples.tolist(),
            "theta_sample_shape": list(theta_samples.shape),
            "metrics": metrics,
            "sweep_reports": sweep_reports,
        },
    }
