from __future__ import annotations

from collections import Counter

from qiskit.primitives import StatevectorSampler, BitArray


def _joint_counts_from_pub_result(pub_res):
    data = pub_res.data

    reg_names = []
    bitarrays = []
    for name in dir(data):
        if name.startswith("_"):
            continue
        obj = getattr(data, name)
        if hasattr(obj, "get_bitstrings") and hasattr(obj, "num_bits"):
            reg_names.append(name)
            bitarrays.append(obj)

    if not bitarrays:
        return {}

    if len(bitarrays) == 1:
        bitstrings = bitarrays[0].get_bitstrings()
        return dict(Counter(bitstrings))

    joint = BitArray.concatenate_bits(bitarrays)
    bitstrings = joint.get_bitstrings()
    return dict(Counter(bitstrings))


def run_ideal_sampler(circuit, shots: int = 1024, seed: int | None = None):
    sampler = StatevectorSampler(seed=seed)
    job = sampler.run([circuit], shots=shots)
    pub_res = job.result()[0]

    counts = _joint_counts_from_pub_result(pub_res)

    try:
        metadata = dict(getattr(pub_res, "metadata", {}) or {})
    except Exception:
        metadata = {}

    return {"counts": counts, "metadata": metadata}