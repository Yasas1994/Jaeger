# Opt-in saving of embedding and NMD vectors — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `jaeger predict` write `_embedding.npz` and `_nmd.npz` only when the user passes `--save-embedding` or `--save-nmd`.

**Architecture:** Add two CLI flags, move the save logic into a small testable helper, gate the `np.savez` calls on the flags, and add unit tests for the helper plus CLI help text.

**Tech Stack:** Python, Click, NumPy, pytest.

---

## File map

| File | Responsibility |
|------|----------------|
| `src/jaeger/cli.py` | Registers `--save-embedding` and `--save-nmd` flags on the `predict` command. |
| `src/jaeger/commands/predict.py` | Helper that writes the `.npz` files only when requested; `run_core` calls it with the flag values. |
| `tests/pytest/test_cli.py` | Asserts that `predict --help` exposes the new flags. |
| `tests/unit/test_commands_predict.py` | Unit tests for the helper: default skips files, flags create correct `.npz` files. |

---

## Task 1: Add CLI flags to `jaeger predict`

**Files:**
- Modify: `src/jaeger/cli.py:265-274`

- [ ] **Step 1: Insert the two new Click options**

Add the options just before the `def predict(**kwargs):` line (after `--onnx` and `--int8`).

```python
@click.option(
    "--save-embedding",
    is_flag=True,
    help="Save per-window embedding vectors to <sample>_embedding.npz",
)
@click.option(
    "--save-nmd",
    is_flag=True,
    help="Save per-window NMD (novelty) vectors to <sample>_nmd.npz",
)
def predict(**kwargs):
```

- [ ] **Step 2: Verify the flags appear in help**

Run:

```bash
python -m pytest tests/pytest/test_cli.py::TestCLICommands::test_predict_command_exists -v
```

This command currently only checks for the word "predict"; we will expand the assertion in Task 3.

- [ ] **Step 3: Commit**

```bash
git add src/jaeger/cli.py
git commit -m "feat(cli): add --save-embedding and --save-nmd flags to predict"
```

---

## Task 2: Gate the vector save logic in `commands/predict.py`

**Files:**
- Modify: `src/jaeger/commands/predict.py:1-19` (imports)
- Modify: `src/jaeger/commands/predict.py:463-476` (replace inline save blocks)
- Create helper in: `src/jaeger/commands/predict.py` near the top, after the `GB_BYTES` constant.

- [ ] **Step 1: Move the NumPy import to the top of the file**

Add `import numpy as np` with the other top-level imports so the helper can use it.

```python
import numpy as np
import psutil
import sys
import time
import traceback
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
import tensorflow as tf
```

Remove the existing `import numpy as np` around line 463.

- [ ] **Step 2: Add the helper function**

Insert after `GB_BYTES = 1024**3`:

```python
def _save_auxiliary_outputs(
    y_pred: dict,
    output_dir: Path,
    file_base: str,
    save_embedding: bool,
    save_nmd: bool,
    logger=None,
) -> None:
    """Write optional embedding and NMD vector files.

    The main ``jaeger predict`` outputs (TSV tables, window scores, FASTA
    sequences, etc.) are handled elsewhere. This helper only persists the
    ``embedding`` and ``nmd`` tensors when the user explicitly requests them.
    """
    if save_embedding and "embedding" in y_pred:
        np.savez(
            output_dir / f"{file_base}_embedding.npz",
            embedding=y_pred["embedding"],
            headers=y_pred["meta_0"],
        )
        if logger is not None:
            logger.info(f"{file_base}_embedding.npz created")
    elif "embedding" in y_pred and logger is not None:
        logger.info("Skipping embedding output; pass --save-embedding to save it.")

    if save_nmd and "nmd" in y_pred:
        np.savez(
            output_dir / f"{file_base}_nmd.npz",
            embedding=y_pred["nmd"],
            headers=y_pred["meta_0"],
        )
        if logger is not None:
            logger.info(f"{file_base}_nmd.npz created")
    elif "nmd" in y_pred and logger is not None:
        logger.info("Skipping nmd output; pass --save-nmd to save it.")
```

- [ ] **Step 3: Replace the inline save blocks with a helper call**

Replace lines 463–476:

```python
        import numpy as np

        if "embedding" in y_pred:
            np.savez(
                OUTPUT_DIR / f"{file_base}_embedding.npz",
                embedding=y_pred["embedding"],
                headers=y_pred["meta_0"],
            )
        if "nmd" in y_pred:
            np.savez(
                OUTPUT_DIR / f"{file_base}_nmd.npz",
                embedding=y_pred["nmd"],
                headers=y_pred["meta_0"],
            )
```

with:

```python
        _save_auxiliary_outputs(
            y_pred,
            OUTPUT_DIR,
            file_base,
            save_embedding=kwargs.get("save_embedding", False),
            save_nmd=kwargs.get("save_nmd", False),
            logger=logger,
        )
```

- [ ] **Step 4: Run a quick import check**

Run:

```bash
python -c "from jaeger.commands.predict import _save_auxiliary_outputs; print('OK')"
```

Expected: prints `OK` with no errors.

- [ ] **Step 5: Commit**

```bash
git add src/jaeger/commands/predict.py
git commit -m "feat(predict): make embedding/nmd saves opt-in via flags"
```

---

## Task 3: Add tests

**Files:**
- Modify: `tests/pytest/test_cli.py`
- Create: `tests/unit/test_commands_predict.py`

- [ ] **Step 1: Add CLI help-text test**

In `tests/pytest/test_cli.py`, add a new test method inside `TestCLICommands`:

```python
    def test_predict_help_shows_save_vector_flags(self):
        """predict --help should advertise --save-embedding and --save-nmd."""
        runner = CliRunner()
        result = runner.invoke(main, ["predict", "--help"])
        assert result.exit_code == 0
        assert "--save-embedding" in result.output
        assert "--save-nmd" in result.output
```

- [ ] **Step 2: Create the helper unit-test file**

Create `tests/unit/test_commands_predict.py`:

```python
"""Tests for jaeger.commands.predict helpers."""

from __future__ import annotations

import numpy as np
from pathlib import Path

from jaeger.commands.predict import _save_auxiliary_outputs


def _fake_y_pred() -> dict[str, np.ndarray]:
    return {
        "prediction": np.zeros((2, 2), dtype=np.float32),
        "embedding": np.ones((2, 8), dtype=np.float32),
        "nmd": np.ones((2, 4), dtype=np.float32),
        "meta_0": np.array(["contig_1", "contig_2"], dtype=object),
    }


def test_default_does_not_save_vectors(tmp_path: Path) -> None:
    y_pred = _fake_y_pred()
    _save_auxiliary_outputs(
        y_pred,
        tmp_path,
        "sample",
        save_embedding=False,
        save_nmd=False,
    )
    assert not (tmp_path / "sample_embedding.npz").exists()
    assert not (tmp_path / "sample_nmd.npz").exists()


def test_save_embedding_flag_writes_file(tmp_path: Path) -> None:
    y_pred = _fake_y_pred()
    _save_auxiliary_outputs(
        y_pred,
        tmp_path,
        "sample",
        save_embedding=True,
        save_nmd=False,
    )
    assert (tmp_path / "sample_embedding.npz").exists()
    assert not (tmp_path / "sample_nmd.npz").exists()

    loaded = np.load(tmp_path / "sample_embedding.npz")
    assert np.array_equal(loaded["embedding"], y_pred["embedding"])
    assert np.array_equal(loaded["headers"], y_pred["meta_0"])


def test_save_nmd_flag_writes_file(tmp_path: Path) -> None:
    y_pred = _fake_y_pred()
    _save_auxiliary_outputs(
        y_pred,
        tmp_path,
        "sample",
        save_embedding=False,
        save_nmd=True,
    )
    assert not (tmp_path / "sample_embedding.npz").exists()
    assert (tmp_path / "sample_nmd.npz").exists()

    loaded = np.load(tmp_path / "sample_nmd.npz")
    assert np.array_equal(loaded["embedding"], y_pred["nmd"])
    assert np.array_equal(loaded["headers"], y_pred["meta_0"])


def test_save_both_flags_writes_both_files(tmp_path: Path) -> None:
    y_pred = _fake_y_pred()
    _save_auxiliary_outputs(
        y_pred,
        tmp_path,
        "sample",
        save_embedding=True,
        save_nmd=True,
    )
    assert (tmp_path / "sample_embedding.npz").exists()
    assert (tmp_path / "sample_nmd.npz").exists()


def test_missing_outputs_are_ignored(tmp_path: Path) -> None:
    """If the model does not expose embedding/nmd, flags must not crash."""
    y_pred = {
        "prediction": np.zeros((2, 2), dtype=np.float32),
        "meta_0": np.array(["contig_1", "contig_2"], dtype=object),
    }
    _save_auxiliary_outputs(
        y_pred,
        tmp_path,
        "sample",
        save_embedding=True,
        save_nmd=True,
    )
    assert not (tmp_path / "sample_embedding.npz").exists()
    assert not (tmp_path / "sample_nmd.npz").exists()
```

- [ ] **Step 3: Run the new tests**

```bash
python -m pytest tests/pytest/test_cli.py::TestCLICommands::test_predict_help_shows_save_vector_flags -v
python -m pytest tests/unit/test_commands_predict.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/pytest/test_cli.py tests/unit/test_commands_predict.py
git commit -m "test(predict): verify embedding/nmd save flags"
```

---

## Task 4: Lint and final verification

- [ ] **Step 1: Run ruff on the changed files**

```bash
ruff check src/jaeger/cli.py src/jaeger/commands/predict.py tests/pytest/test_cli.py tests/unit/test_commands_predict.py
ruff format src/jaeger/cli.py src/jaeger/commands/predict.py tests/pytest/test_cli.py tests/unit/test_commands_predict.py
```

Expected: no lint errors.

- [ ] **Step 2: Run the full unit-test suite**

```bash
python -m pytest tests/unit -q
```

Expected: all tests pass.

- [ ] **Step 3: Commit any formatting fixes**

```bash
git add -u
git commit -m "style: ruff formatting for predict vector flag changes"
```

---

## Spec coverage self-review

| Spec requirement | Implementing task |
|------------------|-------------------|
| Add `--save-embedding` flag | Task 1 |
| Add `--save-nmd` flag | Task 1 |
| Default: do not save either file | Task 2 (`save_embedding=False`, `save_nmd=False`) |
| Keep file naming / contents unchanged when requested | Task 2 (same `np.savez` contents) |
| Info log when skipping | Task 2 (`logger.info` in `elif` branches) |
| CLI help exposes flags | Task 3 test |
| Unit tests for default and flag-enabled behaviour | Task 3 tests |

No placeholders or undefined names remain in the plan.
