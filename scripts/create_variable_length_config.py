#!/usr/bin/env python3
"""Create a dynamic-length training config from a fixed-length base config."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-suffix", default="variable")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.base_config.read_text())
    model = cfg.setdefault("model", {})

    old_name = model.get("name", "jaeger")
    old_exp = model.get("experiment", "experiment")
    suffix = args.experiment_suffix

    model["name"] = f"{old_name.rsplit('_', 1)[0]}_{suffix}"
    model["experiment"] = f"{old_exp}_{suffix}"

    sp = model.setdefault("string_processor", {})
    sp["crop_size"] = None
    sp["length"] = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.dump(cfg))
    print(f"Wrote variable-length config to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
