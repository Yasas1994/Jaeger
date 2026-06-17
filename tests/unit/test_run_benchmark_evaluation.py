# tests/unit/test_run_benchmark_evaluation.py
import csv
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.run_benchmark_evaluation import main


def test_run_benchmark_evaluation(tmp_path):
    experiments = tmp_path / "experiments"
    data = tmp_path / "data"
    out_csv = tmp_path / "evaluation_metrics.csv"

    exp = experiments / "experiment_500bp_baseline_trans_42"
    graph = exp / "jaeger_500bp_baseline_trans_graph"
    graph.mkdir(parents=True)
    (graph / "saved_model.pb").write_text("fake")

    data.mkdir(parents=True)
    val = data / "val_shuffled_translated_500.npz"
    val.write_text("fake")

    def fake_subprocess(cmd, *args, **kwargs):
        out = Path(cmd[cmd.index("--output-csv") + 1])
        cm_out = Path(cmd[cmd.index("--output-cm") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        cm_out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "model_dir,npz,num_samples,overall_accuracy,balanced_accuracy\n"
            f"{graph},{val},100,0.6,0.5\n"
        )
        np.save(cm_out, np.array([[30, 5, 5], [2, 20, 8], [3, 7, 20]]))
        class Proc:
            returncode = 0
        return Proc()

    with patch("subprocess.run", side_effect=fake_subprocess):
        old_argv = sys.argv
        try:
            sys.argv = [
                "run_benchmark_evaluation.py",
                "--experiments-root", str(experiments),
                "--data-root", str(data),
                "--output-csv", str(out_csv),
            ]
            main()
        finally:
            sys.argv = old_argv

    with open(out_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["experiment"] == "experiment_500bp_baseline_trans_42"
    assert float(rows[0]["overall_accuracy"]) == 0.6
    assert Path(rows[0]["cm_path"]).exists()
