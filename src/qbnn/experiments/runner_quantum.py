
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
)
from src.qbnn.models import build_bayesian_model
from src.qbnn.partition import build_partition_blocks
from src.qbnn.quantum.oracles import build_local_log_posterior
from src.qbnn.quantum.circuits import (
    build_coherent_mh_problem,
    build_coherent_metropolis_move_problem,
    build_szegedy_qpe_problem,
)
from src.qbnn.quantum.execution import (
    run_ideal_sampler,
    run_noisy_sampler,
    build_local_fake_backend,
    build_service,
    get_backend,
    run_ibm_sampler,
)
from src.qbnn.quantum.posterior_sampling import (
    state_probs_from_counts,
    szegedy_zero_phase_state_probs,
    sample_state_indices,
    embed_local_samples,
)
from src.qbnn.quantum.evaluate import evaluate_theta_samples, distribution_diagnostics
from src.qbnn.quantum.resources import logical_resource_report, transpiled_resource_report
from src.qbnn.proposals import (
    build_signed_direction_moves,
    build_hidden_pathway_moves,
    build_output_bias_moves,
)
from src.qbnn.quantum.oracles.local_delta_oracle import (
    build_local_move_table,
    build_hidden_pathway_move_table,
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

    if fam == "coherent_metropolis_move":
        raise RuntimeError("coherent_metropolis_move should use _diagnose_move_block, not _diagnose_block")

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

    if cfg.partition.strategy in {"region_aligned_weight_blocks", "hidden_pathway_blocks"}:
        return build_partition_blocks(
            model.num_params,
            cfg.partition.strategy,
            block_param_count=cfg.partition.block_param_count,
            model=model,
        )

    return build_partition_blocks(
        model.num_params,
        cfg.partition.strategy,
        block_param_count=cfg.partition.block_param_count,
    )


def _decode_move_counts(counts: dict, dir_bits: int, actual_num_moves: int) -> dict:
    out: dict[int | str, float] = {}
    total = float(sum(counts.values())) if counts else 0.0
    if total <= 0:
        return out

    for key, value in counts.items():
        bits = str(key).replace(" ", "")
        bits = bits.zfill(dir_bits + 2)

        accepted, move_idx = decode_move_bitstring(bits, dir_bits)
        prob = float(value) / total

        if (not accepted) or (move_idx < 0) or (move_idx >= actual_num_moves):
            out["reject"] = out.get("reject", 0.0) + prob
        else:
            out[move_idx] = out.get(move_idx, 0.0) + prob

    return out


def _sample_move_from_probs(move_probs: dict, rng: np.random.Generator):
    items = list(move_probs.items())
    labels = [k for k, _ in items]
    probs = np.asarray([p for _, p in items], dtype=np.float64)
    probs = probs / probs.sum()
    idx = int(rng.choice(len(labels), p=probs))
    return labels[idx]


def _apply_sampled_move(theta, active, move_table, move_choice):
    theta_new = np.asarray(theta, dtype=np.float64).copy()
    active = np.asarray(active, dtype=np.int64)

    if move_choice is None:
        return theta_new

    if isinstance(move_choice, str):
        if move_choice.lower() == "reject":
            return theta_new
        try:
            move_idx = int(move_choice)
        except ValueError:
            return theta_new
    else:
        try:
            move_idx = int(move_choice)
        except (TypeError, ValueError):
            return theta_new

    proposal_states = np.asarray(move_table.proposal_states, dtype=np.float64)
    num_actual_moves = int(proposal_states.shape[0])

    if move_idx < 0 or move_idx >= num_actual_moves:
        return theta_new

    theta_new[active] = proposal_states[move_idx]
    return theta_new


def _representative_only(items: list[dict], keep: int) -> list[dict]:
    return items[:keep]


def _is_output_bias_block(active: np.ndarray, model) -> bool:
    active = np.asarray(active, dtype=np.int64)
    return np.all((active >= int(model.s_fc2_b.start)) & (active < int(model.s_fc2_b.stop)))


def _build_move_table(cfg: ExperimentConfig, model, theta_ref: np.ndarray, active: np.ndarray, data: dict):
    move_family = str(cfg.quantum.extra.get("move_family", "direction_bank")).strip().lower()

    if move_family == "hidden_pathway":
        if _is_output_bias_block(active, model):
            bias_step = float(cfg.quantum.extra.get("output_bias_step", cfg.quantum.extra.get("direction_step", 0.025)))
            moves = build_output_bias_moves(
                model,
                active,
                theta_ref,
                bias_step=bias_step,
            )
            move_table = build_local_move_table(
                model,
                theta_ref,
                active,
                moves.proposal_states,
                moves.move_labels,
                data["x_train"],
                data["y_train"],
            )
            setattr(move_table, "num_directions", int(moves.num_directions))
            return move_table, {"move_family": "output_bias", "num_directions": int(moves.num_directions)}

        step_in = float(cfg.quantum.extra.get("hidden_step_in", cfg.quantum.extra.get("direction_step", 0.025)))
        step_out = float(cfg.quantum.extra.get("hidden_step_out", step_in))
        bias_step = float(cfg.quantum.extra.get("hidden_bias_step", step_in))
        output_mode = str(cfg.quantum.extra.get("output_mode", "zero_sum_one_vs_rest")).strip()
        output_reference_class = int(cfg.quantum.extra.get("output_reference_class", 0))
        output_pattern_count_raw = cfg.quantum.extra.get("output_pattern_count")
        output_pattern_count = (
            None
            if output_pattern_count_raw in {None, "", "null", "none"}
            else int(output_pattern_count_raw)
        )


        moves = build_hidden_pathway_moves(
            model,
            active,
            theta_ref,
            step_in=step_in,
            step_out=step_out,
            bias_step=bias_step,
            output_mode=str(cfg.quantum.extra.get("output_mode", "zero_sum_one_vs_rest")),
            output_reference_class=int(cfg.quantum.extra.get("output_reference_class", 0)),
            output_pattern_count=cfg.quantum.extra.get("output_pattern_count", None),
        )
        move_table = build_hidden_pathway_move_table(
            model,
            theta_ref,
            active,
            moves.proposal_states,
            moves.move_labels,
            data["x_train"],
            data["y_train"],
        )
        setattr(move_table, "num_directions", int(moves.num_directions))
        return move_table, {"move_family": "hidden_pathway", "num_directions": int(moves.num_directions)}

    direction_step = float(cfg.quantum.extra.get("direction_step", 0.025))
    direction_bank_size = int(cfg.quantum.extra.get("direction_bank_size", 16))
    moves = build_signed_direction_moves(
        active,
        theta_ref,
        direction_bank_size=direction_bank_size,
        direction_step=direction_step,
    )
    move_table = build_local_move_table(
        model,
        theta_ref,
        active,
        moves.proposal_states,
        moves.move_labels,
        data["x_train"],
        data["y_train"],
    )
    setattr(move_table, "num_directions", int(moves.num_directions))
    return move_table, {"move_family": "direction_bank", "num_directions": int(moves.num_directions)}


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
    codec = FixedPointCodec(bits=int(cfg.quantum.extra.get("discretization_bits", 1)), step=float(cfg.quantum.extra.get("quant_step", 0.05)))
    proposal_kind = str(cfg.quantum.extra.get("proposal_kind", "hamming"))
    blocks = _build_blocks(cfg, model)
    backend = _get_backend_for_reports(cfg)
    max_blocks = int(cfg.quantum.extra.get("diagnostic_block_count", 3))
    block_reports = []

    for block_id, active in enumerate(blocks[:max_blocks]):
        if cfg.quantum.family == "coherent_metropolis_move":
            move_table, move_meta = _build_move_table(cfg, model, theta_ref, active, data)

            rep = _diagnose_move_block(
                cfg,
                move_table,
                backend,
                shots=min(cfg.sampling.exploratory_shots, 256),
                seed=cfg.sampling.random_seed + block_id,
            )
            rep["block_id"] = block_id
            rep["active_indices"] = active.tolist()
            rep["block_size"] = int(active.size)
            rep["num_moves"] = int(move_table.accept_probs.size)
            rep["move_family"] = move_meta["move_family"]
            block_reports.append(rep)
            continue

        state_space = build_local_state_space(active, theta_ref, codec)
        log_pi = build_local_log_posterior(model, theta_ref, active, state_space.states, data["x_train"], data["y_train"])

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
            cfg, p, current_state, backend,
            shots=min(cfg.sampling.exploratory_shots, 256),
            seed=cfg.sampling.random_seed + block_id,
            target_pi=pi,
        )
        rep["block_id"] = block_id
        rep["active_indices"] = active.tolist()
        rep["state_space_size"] = int(state_space.num_states)
        rep["detailed_balance_error"] = detailed_balance_error(p, pi)
        rep["target_stationary_distribution"] = pi.tolist()
        rep["distribution_metrics"] = distribution_diagnostics(rep.get("empirical_state_probs", {}), pi) if rep.get("empirical_state_probs") else None
        block_reports.append(rep)

    return {"config": asdict(cfg), "diagnostics": block_reports}


def run_quantum_experiment(cfg: ExperimentConfig) -> dict:
    data = load_zipcode_dataset(cfg.data)
    model = build_bayesian_model(cfg.model)
    theta = _load_reference_theta(cfg.quantum.extra.get("reference_theta_path"), model.num_params)
    codec = FixedPointCodec(bits=int(cfg.quantum.extra.get("discretization_bits", 1)), step=float(cfg.quantum.extra.get("quant_step", 0.05)))
    proposal_kind = str(cfg.quantum.extra.get("proposal_kind", "hamming"))
    blocks = _build_blocks(cfg, model)
    print("num blocks =", len(blocks))
    print("block sizes =", [len(b) for b in blocks])
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
            if cfg.quantum.family == "coherent_metropolis_move":
                move_table, move_meta = _build_move_table(cfg, model, theta, active, data)

                if sweep == 0 and block_id == 0:
                    print("\n=== DEBUG FIRST BLOCK MOVE TABLE ===")
                    print(f"active size = {len(active)}")
                    print(f"num moves = {len(move_table.proposal_states)}")
                    print(f"log_pi_current = {float(move_table.log_pi_current):.8f}")
                    k = min(8, len(move_table.proposal_states))
                    for i in range(k):
                        print(
                            f"move {i:02d} | "
                            f"log_pi_proposed = {float(move_table.log_pi_proposals[i]):.8f} | "
                            f"accept_prob = {float(move_table.accept_probs[i]):.8f}"
                        )

                diag = _diagnose_move_block(
                    cfg,
                    move_table,
                    backend,
                    shots=shots,
                    seed=cfg.sampling.random_seed + 97 * sweep + block_id,
                )

                if sweep == 0 and block_id == 0:
                    print("\n=== DEBUG FIRST BLOCK COUNTS ===")
                    print(diag.get("sample_counts"))

                move_probs = diag.get("empirical_move_probs", {})
                if not move_probs:
                    raise RuntimeError(f"No empirical move probabilities recovered for block {block_id}")
                move_choice = _sample_move_from_probs(move_probs, rng)


                info = {
                    "block_id": block_id,
                    "active_indices": active.tolist(),
                    "block_size": int(active.size),
                    "num_moves": int(move_table.accept_probs.size),
                    "shots": int(shots),
                    "move_choice": move_choice,
                    "move_family": move_meta["move_family"],
                    "logical": diag.get("logical"),
                }
                if block_id < diagnostic_block_limit:
                    info["sample_resources"] = diag.get("sample_resources")
                    info["transpiled_sample_resources"] = diag.get("transpiled_sample_resources")
                per_block.append(info)
                continue

            state_space = build_local_state_space(active, theta, codec)
            log_pi = build_local_log_posterior(model, theta, active, state_space.states, data["x_train"], data["y_train"])

            if not np.all(np.isfinite(log_pi)):
                raise ValueError(
                    f"Non-finite log_pi at sweep={sweep}, block={block_id}, active={active.tolist()}, log_pi={log_pi}"
                )

            q = _proposal(proposal_kind, state_space.states)
            p, pi = build_mh_transition_matrix(log_pi, q)

            if not np.all(np.isfinite(p)):
                raise ValueError(
                    f"Non-finite transition matrix at sweep={sweep}, block={block_id}, active={active.tolist()}, p={p}, log_pi={log_pi}"
                )
            current_state = _current_local_state_index(state_space.states, theta[active])

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


def _diagnose_move_block(cfg: ExperimentConfig, move_table, backend_for_reports, shots: int, seed: int) -> dict:
    num_dirs = int(getattr(move_table, "num_directions", 0) or cfg.quantum.extra.get("direction_bank_size", 16))
    problem = build_coherent_metropolis_move_problem(
        move_table.accept_probs,
        num_directions=num_dirs,
    )
    sample_counts = _run_counts(
        cfg,
        problem.sample_circuit,
        shots=shots,
        seed=seed,
        backend=backend_for_reports,
    )

    actual_num_moves = int(problem.logical_info.get("actual_num_moves", move_table.accept_probs.size))
    move_probs = _decode_move_counts(
        sample_counts,
        int(problem.logical_info["dir_bits"]),
        actual_num_moves,
    )

    diag = {
        "family": "coherent_metropolis_move",
        "logical": problem.logical_info,
        "sample_counts": sample_counts,
        "empirical_move_probs": move_probs,
        "sample_resources": logical_resource_report(problem.sample_circuit),
        "qpe_counts": None,
    }
    if backend_for_reports is not None:
        diag["transpiled_sample_resources"] = transpiled_resource_report(
            problem.sample_circuit,
            backend_for_reports,
            optimization_level=cfg.quantum.optimization_level,
        )
    return diag


def decode_move_bitstring(bitstring: str, dir_bits: int):
    if len(bitstring) != dir_bits + 2:
        raise ValueError(f"expected bitstring length {dir_bits + 2}, got {len(bitstring)}")

    acc_bit = bitstring[0]
    sign_bit = bitstring[1]
    dir_str = bitstring[2:]

    accepted = (acc_bit == "1")
    direction = int(dir_str, 2) if dir_str else 0
    sign = int(sign_bit)

    move_idx = 2 * direction + sign
    return accepted, move_idx
