from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    carrier_frequency_hz: float
    target_frequency_hz: float
    conduction_velocity_m_s: float
    spatial_decay_lambda_m: float
    receptor_spacing_m: float
    receptor_padding_m: float
    bandpass_low_hz: float
    bandpass_high_hz: float
    bandpass_order: int
    membrane_tau_s: float
    refractory_s: float
    threshold: float
    gain: float
    smoothing_sigma: float
    clarity_mass_fraction: float
    enable_gpu: bool
    gpu_device_id: int
    chunk_size: int
    receptor_chunk_size: int
    float_dtype: str
    steady_state_only: bool
    fft_window_cycles: int


@dataclass
class DataConfig:
    mat_file: str
    required_methods: list[str]


@dataclass
class OutputConfig:
    results_dir: str
    save_intermediate_npz: bool
    save_population_maps: bool
    save_spike_trains: bool


@dataclass
class ExperimentConfig:
    name: str
    task_names: dict[str, str]
    logistic_scale: float
    standardize_scores: bool
    pairwise_methods: list[str]


@dataclass
class FullConfig:
    root_dir: Path
    model: ModelConfig
    data: DataConfig
    output: OutputConfig
    experiment: ExperimentConfig

    @property
    def mat_path(self) -> Path:
        return (self.root_dir / self.data.mat_file).resolve()

    @property
    def results_dir(self) -> Path:
        return (self.root_dir / self.output.results_dir).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(model_config_path: str | Path, experiment_config_path: str | Path) -> FullConfig:
    model_path = Path(model_config_path).resolve()
    exp_path = Path(experiment_config_path).resolve()
    root_dir = model_path.parent

    model_raw = _load_yaml(model_path)
    exp_raw = _load_yaml(exp_path)

    return FullConfig(
        root_dir=root_dir,
        model=ModelConfig(**model_raw["model"]),
        data=DataConfig(**model_raw["data"]),
        output=OutputConfig(**model_raw["output"]),
        experiment=ExperimentConfig(**exp_raw["experiment"]),
    )
