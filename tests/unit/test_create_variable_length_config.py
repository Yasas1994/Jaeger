# tests/unit/test_create_variable_length_config.py
import sys

import yaml

from scripts.create_variable_length_config import main


def test_create_variable_length_config(tmp_path):
    base = tmp_path / "base.yaml"
    out = tmp_path / "out.yaml"

    cfg = {
        "model": {
            "name": "jaeger_500bp_baseline_trans",
            "experiment": "500bp_baseline_trans",
            "base_dir": "/tmp",
            "string_processor": {"crop_size": 500, "length": 500},
        }
    }
    base.write_text(yaml.dump(cfg, sort_keys=False))

    old_argv = sys.argv
    try:
        sys.argv = [
            "create_variable_length_config.py",
            "--base-config", str(base),
            "--output", str(out),
            "--experiment-suffix", "variable",
        ]
        main()
    finally:
        sys.argv = old_argv

    result = yaml.safe_load(out.read_text())
    assert result["model"]["name"] == "jaeger_500bp_baseline_variable"
    assert result["model"]["experiment"] == "500bp_baseline_trans_variable"
    assert result["model"]["string_processor"]["crop_size"] is None
    assert result["model"]["string_processor"]["length"] is None
