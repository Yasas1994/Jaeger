"""Regression tests: prophage plots must map window indices with the stride
and must not draw past the contig end.

Windows are stride-spaced (overlapping when stride < fsize), so the x-axis is
``window_index * stride``. The last prediction window of a contig is partial,
so its x can exceed the true contig length; pycirclize rejects any x beyond
the sector size (crash: "x=... is invalid range of '...' sector").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jaeger.postprocess.prophages import (
    logits_to_df_v2,
    plot_scores,
    plot_scores_linear,
)

CLASS_MAP = {"class": ["bacteria", "phage"], "index": [0, 1]}
CONFIG = {"all_labels": {0: "bacteria", 1: "phage"}}


def _logits_df(n_windows: int, contig_len: int, stride: int = 2000) -> dict:
    df = pd.DataFrame(
        {
            "bacteria": np.linspace(0.1, 0.4, n_windows),
            "phage": np.linspace(3.5, 0.5, n_windows),
            # clamped x-axis, as produced by logits_to_df_v2
            "length": [min(i * stride, contig_len) for i in range(n_windows)],
            "gc": np.zeros(n_windows),
            "gc_skew": np.zeros(n_windows),
        }
    )
    return {"contig1": [df, "bacteria", contig_len]}


def _predictions(n_windows: int) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.normal(size=(n_windows, 2))]


def test_logits_to_df_v2_clamps_terminal_window_x():
    # 4 windows at fsize=2000 (no stride -> stride=fsize) -> unclamped x
    # would be [0, 2000, 4000, 6000] for a 4857 bp contig
    out = logits_to_df_v2(
        class_map=CLASS_MAP,
        cmdline_kwargs={"fsize": 2000, "lc": 0},
        headers=np.array(["contig1"]),
        predictions=_predictions(4),
        lengths=np.array([4857]),
        gc_skews=[np.zeros(4)],
        gcs=[np.zeros(4)],
    )
    df, _, _ = out["contig1"]
    assert df["length"].tolist() == [0, 2000, 4000, 4857]


def test_logits_to_df_v2_uses_stride_for_x_axis():
    # stride=1500: window starts at i*1500, not i*2000
    out = logits_to_df_v2(
        class_map=CLASS_MAP,
        cmdline_kwargs={"fsize": 2000, "stride": 1500, "lc": 0},
        headers=np.array(["contig1"]),
        predictions=_predictions(5),
        lengths=np.array([4857]),
        gc_skews=[np.zeros(5)],
        gcs=[np.zeros(5)],
    )
    df, _, _ = out["contig1"]
    assert df["length"].tolist() == [0, 1500, 3000, 4500, 4857]


def test_circular_plot_tolerates_terminal_overshoot(tmp_path):
    contig_len = 4857
    logits_df = _logits_df(n_windows=4, contig_len=contig_len)
    # prophage range spanning all 4 windows -> highlight end (3*2000+2000)
    # overshoots the contig
    phage_cordinates = {"contig1": [np.array([[0, 4]]), np.array([2.5])]}
    plot_scores(
        logits_df,
        config=CONFIG,
        model="test",
        fsize=2000,
        infile_base="test",
        outdir=tmp_path,
        phage_cordinates=phage_cordinates,
    )
    assert (tmp_path / "test_jaeger_contig1.pdf").exists()


def test_circular_plot_stride_overshoot(tmp_path):
    contig_len = 4857
    logits_df = _logits_df(n_windows=4, contig_len=contig_len, stride=1500)
    phage_cordinates = {"contig1": [np.array([[0, 4]]), np.array([2.5])]}
    plot_scores(
        logits_df,
        config=CONFIG,
        model="test",
        fsize=2000,
        infile_base="test",
        outdir=tmp_path,
        phage_cordinates=phage_cordinates,
        stride=1500,
    )
    assert (tmp_path / "test_jaeger_contig1.pdf").exists()


def test_linear_plot_tolerates_terminal_overshoot(tmp_path):
    contig_len = 4857
    logits_df = _logits_df(n_windows=4, contig_len=contig_len)
    phage_cordinates = {"contig1": [np.array([[0, 4]]), np.array([2.5])]}
    plot_scores_linear(
        logits_df,
        config=CONFIG,
        model="test",
        fsize=2000,
        infile_base="test",
        outdir=tmp_path,
        phage_cordinates=phage_cordinates,
        stride=1500,
    )
    assert (tmp_path / "test_jaeger_contig1_linear.pdf").exists()
