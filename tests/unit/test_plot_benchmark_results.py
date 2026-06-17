# tests/unit/test_plot_benchmark_results.py
import sys

import numpy as np

from scripts.plot_benchmark_results import main


def test_plot_benchmark_results(tmp_path):
    report = tmp_path / "benchmark_report.csv"
    metrics = tmp_path / "evaluation_metrics.csv"
    out_dir = tmp_path / "figures"

    cm_trans = tmp_path / "cm_trans.npy"
    cm_nuc = tmp_path / "cm_nuc.npy"
    np.save(cm_trans, np.array([[30, 5, 5], [2, 20, 8], [3, 7, 20]]))
    np.save(cm_nuc, np.array([[25, 10, 5], [5, 15, 10], [5, 10, 15]]))

    report.write_text(
        "experiment,epochs_trained,best_epoch,best_val_accuracy,best_loss_epoch,\n"
        "experiment_500bp_baseline_trans_42,5,4,0.55,3,\n"
        "experiment_500bp_baseline_nuc_42,5,4,0.50,3,\n"
    )
    metrics.write_text(
        "experiment,length_bp,input_type,cm_path,model_dir,npz,num_samples,overall_accuracy,balanced_accuracy,f1_class_0,f1_class_1,f1_class_2\n"
        f"experiment_500bp_baseline_trans_42,500,translated,{cm_trans},m,n,100,0.55,0.50,0.9,0.4,0.3\n"
        f"experiment_500bp_baseline_nuc_42,500,nucleotide,{cm_nuc},m,n,100,0.50,0.45,0.85,0.35,0.25\n"
    )

    old_argv = sys.argv
    try:
        sys.argv = [
            "plot_benchmark_results.py",
            "--report-csv", str(report),
            "--metrics-csv", str(metrics),
            "--output-dir", str(out_dir),
        ]
        main()
    finally:
        sys.argv = old_argv

    assert (out_dir / "f1_per_class_bar.png").exists()
    assert (out_dir / "accuracy_vs_length.png").exists()
    assert (out_dir / "confusion_matrix_grid.png").exists()
    assert (out_dir / "training_curves.png").exists()
