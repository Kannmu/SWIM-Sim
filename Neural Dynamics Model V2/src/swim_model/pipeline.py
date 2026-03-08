from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .backend import ArrayBackend, gpu_available
from .config import load_config
from .io.load_kwave_mat import KWaveMatLoader
from .neural.population_simulator import PopulationSimulator
from .preprocessing.receptor_lattice import build_receptor_lattice
from .readout.clarity_score import compute_clarity_score
from .readout.intensity_score import compute_intensity_score
from .readout.pairwise_prediction import pairwise_preferences


def _select_backend(cfg):
    return ArrayBackend(use_gpu=bool(cfg.model.enable_gpu and gpu_available()))


def run_full_pipeline(model_config_path, experiment_config_path):
    cfg = load_config(model_config_path, experiment_config_path)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    loader = KWaveMatLoader(cfg.mat_path)
    data = loader.load()

    method_names = [m for m in cfg.data.required_methods if m in data["methods"]]
    if not method_names:
        raise ValueError("No required methods were found in the K-Wave data file.")

    reference = data["methods"][method_names[0]]
    lattice = build_receptor_lattice(
        reference["roi_x"],
        reference["roi_y"],
        spacing_m=cfg.model.receptor_spacing_m,
        padding_m=cfg.model.receptor_padding_m,
    )

    backend = _select_backend(cfg)
    simulator = PopulationSimulator(cfg, backend)

    condition_results = {}
    intensity_scores = {}
    clarity_scores = {}

    for method_name in method_names:
        result = simulator.run_condition(method_name, data["methods"][method_name], lattice)
        intensity = compute_intensity_score(result["weights"])
        clarity, area90 = compute_clarity_score(
            result["population_map"],
            result["roi_x"],
            result["roi_y"],
            mass_fraction=cfg.model.clarity_mass_fraction,
            sigma=cfg.model.smoothing_sigma,
        )
        result["intensity_score"] = intensity
        result["clarity_score"] = clarity
        result["a90_area_m2"] = area90
        condition_results[method_name] = result
        intensity_scores[method_name] = intensity
        clarity_scores[method_name] = clarity

        if cfg.output.save_intermediate_npz:
            np.savez_compressed(
                cfg.results_dir / f"{method_name}_population_outputs.npz",
                weights=result["weights"],
                rates=result["rates"],
                vector_strength=result["vector_strength"],
                population_map=result["population_map"],
                receptor_coords_m=result["receptor_coords_m"],
            )
        if cfg.output.save_spike_trains:
            np.save(cfg.results_dir / f"{method_name}_spikes.npy", result["spikes"].astype(np.uint8))

    intensity_z, intensity_pairwise = pairwise_preferences(
        {m: intensity_scores[m] for m in cfg.experiment.pairwise_methods if m in intensity_scores},
        logistic_scale=cfg.experiment.logistic_scale,
        standardize=cfg.experiment.standardize_scores,
    )
    clarity_z, clarity_pairwise = pairwise_preferences(
        {m: clarity_scores[m] for m in cfg.experiment.pairwise_methods if m in clarity_scores},
        logistic_scale=cfg.experiment.logistic_scale,
        standardize=cfg.experiment.standardize_scores,
    )

    summary = {
        "backend": "gpu" if backend.use_gpu else "cpu",
        "n_receptors": int(len(lattice["coords_m"])),
        "methods": {
            method_name: {
                "intensity_score": float(condition_results[method_name]["intensity_score"]),
                "clarity_score": float(condition_results[method_name]["clarity_score"]),
                "a90_area_m2": float(condition_results[method_name]["a90_area_m2"]),
                "mean_rate_hz": float(np.mean(condition_results[method_name]["rates"])),
                "mean_vector_strength": float(np.mean(condition_results[method_name]["vector_strength"])),
            }
            for method_name in method_names
        },
        "intensity_zscore": intensity_z,
        "clarity_zscore": clarity_z,
        "pairwise": {
            "intensity": intensity_pairwise,
            "clarity": clarity_pairwise,
        },
    }

    with (cfg.results_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
