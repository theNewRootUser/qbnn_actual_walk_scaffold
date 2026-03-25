from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class DataConfig:
    dataset_name: str = "zipcode"
    npz_path: Optional[str] = "data/zipcode/zipcode.npz"
    train_path: Optional[str] = None
    test_path: Optional[str] = None
    random_seed: int = 42
    standardize: bool = False
    image_height: int = 16
    image_width: int = 16
    num_channels: int = 1
    num_classes: int = 10

@dataclass
class ModelConfig:
    architecture: str = "lenet2"
    prior_std: float = 1.0
    image_height: int = 16
    image_width: int = 16
    num_channels: int = 1
    num_classes: int = 10
    conv1_out: int = 12
    conv2_out: int = 6
    fc_hidden: int = 30

@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    random_seed: int = 42
    checkpoint_json_path: Optional[str] = "outputs/classical_lenet2_zipcode.json"

@dataclass
class SamplingConfig:
    sweeps_total: int = 8
    final_sweeps: int = 1
    exploratory_shots: int = 256
    final_shots: int = 1024
    theta_samples_per_block: int = 1
    random_seed: int = 1234
    save_all_sweeps: bool = True
    score_only_final_sweeps: bool = True

@dataclass
class QuantumConfig:
    pe_mode: str = "qpe"            # qpe | rall
    execution_mode: str = "ideal"   # ideal | noisy | ibm
    family: str = "coherent_mh"     # coherent_mh | szegedy | coherent_metropolis_move
    num_eval_qubits: int = 4
    optimization_level: int = 1
    backend_name: Optional[str] = None
    service_channel: Optional[str] = None
    instance: Optional[str] = None
    noise_backend_name: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PartitionConfig:
    enabled: bool = True
    strategy: str = "contiguous_weight_blocks"
    block_param_count: int = 1
    explicit_blocks: Optional[List[List[int]]] = None

@dataclass
class ExperimentConfig:
    name: str = "lenet2_zipcode"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    partition: PartitionConfig = field(default_factory=PartitionConfig)
