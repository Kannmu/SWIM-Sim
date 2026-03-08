from __future__ import annotations

import json
from pathlib import Path

from src.swim_model.pipeline import run_full_pipeline


def main():
    root = Path(__file__).resolve().parent
    summary = run_full_pipeline(
        root / "configs" / "pacinian_model.yaml",
        root / "configs" / "experiment1.yaml",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
