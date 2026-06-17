# tests/unit/test_merge_npz_for_variable_length.py
import sys

import numpy as np

from scripts.merge_npz_for_variable_length import main


def test_merge_npz(tmp_path):
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    out = tmp_path / "mixed.npz"

    np.savez(a, features=np.arange(6).reshape(2, 3), labels=np.array([0, 1]))
    np.savez(b, features=np.arange(12).reshape(2, 6), labels=np.array([1, 2]))

    old_argv = sys.argv
    try:
        sys.argv = [
            "merge_npz_for_variable_length.py",
            "--inputs", str(a), str(b),
            "--output", str(out),
        ]
        main()
    finally:
        sys.argv = old_argv

    data = np.load(out)
    assert data["features"].shape[0] == 4
    assert data["labels"].shape[0] == 4
