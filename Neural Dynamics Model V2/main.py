from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.swim_model.pipeline import run_full_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run the SWIM neural dynamics model.")
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment1",
        choices=["experiment1", "supp1"],
        help="Which experiment to run: 'experiment1' (default) or 'supp1' (supplementary experiment 1).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent

    if args.experiment == "experiment1":
        model_config = root / "configs" / "pacinian_model.yaml"
        experiment_config = root / "configs" / "experiment1.yaml"
    elif args.experiment == "supp1":
        model_config = root / "configs" / "pacinian_model_supp1.yaml"
        experiment_config = root / "configs" / "supplementary_experiment1.yaml"
    else:
        # Should be covered by argparse choices, but for safety
        raise ValueError(f"Unknown experiment: {args.experiment}")

    summary = run_full_pipeline(
        model_config,
        experiment_config,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
