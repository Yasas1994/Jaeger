import numpy as np
import polars as pl
import pytest
from pathlib import Path

from jaeger.postprocess import refinement as rf


def _make_logit_df(logits: np.ndarray, true: list[str] | None = None) -> pl.DataFrame:
    df = pl.DataFrame(
        {score_col: logits[:, i] for i, score_col in enumerate(rf.SCORE_COLS)}
    )
    predictions = logits.argmax(axis=1)
    df = df.with_columns(
        pl.Series("prediction", [rf.CLASSES[i] for i in predictions]),
    )
    if true is not None:
        df = df.with_columns(pl.Series("true", true))
    return df


def test_add_score_features_basic():
    # phage high, everything else low.
    logits = np.zeros((1, 6), dtype=np.float32)
    logits[0, 0] = 3.0  # phage
    logits[0, 1] = 1.0  # virus
    df = _make_logit_df(logits)
    df = rf.add_score_features(df)

    assert df["top_class"][0] == "phage"
    assert df["second_class"][0] == "virus"
    assert df["top_logit"][0] == pytest.approx(3.0)
    assert df["margin"][0] == pytest.approx(2.0)
    assert 0.0 < df["top_prob"][0] <= 1.0
    assert df["entropy"][0] >= 0.0


def test_fit_thresholds_uses_correct_diagonal():
    # 30 windows (above the sample-size fallback) all correctly predicted as
    # bacteria with logit 2.0 and runner-up plasmid at 0.5.
    logits = np.zeros((30, 6), dtype=np.float32)
    logits[:, 3] = 2.0  # bacteria
    logits[:, 4] = 0.5  # plasmid runner-up
    true = ["bacteria"] * 30
    df = _make_logit_df(logits, true)
    df = rf.add_score_features(df)

    taus = rf.fit_thresholds(df, quantile=0.0)
    # 0th percentile should equal the minimum observed value.
    assert taus["bacteria"]["logit"] == pytest.approx(2.0)
    assert taus["bacteria"]["margin"] == pytest.approx(1.5)


def test_fit_thresholds_falls_back_with_few_samples():
    logits = np.zeros((1, 6), dtype=np.float32)
    logits[0, 3] = 2.0
    df = _make_logit_df(logits, ["bacteria"])
    df = rf.add_score_features(df)
    taus = rf.fit_thresholds(df, quantile=0.05)
    assert taus["bacteria"]["logit"] == -np.inf
    assert taus["bacteria"]["margin"] == -np.inf


def test_refine_merges_before_abstain():
    # bacteria vs plasmid with low margin should become bacteria_or_plasmid,
    # not unknown, even though top_logit may be below a strict threshold.
    logits = np.zeros((1, 6), dtype=np.float32)
    logits[0, 3] = 0.6  # bacteria
    logits[0, 4] = 0.4  # plasmid
    df = _make_logit_df(logits)
    df = rf.add_score_features(df)

    taus = {c: {"logit": 1.0, "margin": 0.5} for c in rf.CLASSES}
    refined = rf.refine(df, taus)
    assert refined["refined_prediction"][0] == "bacteria_or_plasmid"


def test_refine_abstains_low_confidence():
    # Low-confidence bacteria over eukarya (not a merge pair) should abstain.
    logits = np.zeros((1, 6), dtype=np.float32)
    logits[0, 3] = 0.3  # bacteria
    logits[0, 5] = 0.1  # eukarya runner-up
    df = _make_logit_df(logits)
    df = rf.add_score_features(df)

    taus = {c: {"logit": 1.0, "margin": 0.5} for c in rf.CLASSES}
    refined = rf.refine(df, taus)
    assert refined["refined_prediction"][0] == "unknown"


def test_aggregate_contig_gated_drops_unknown():
    # Two windows: one confident phage, one unknown.
    rows = []
    for i, (cid, scores, refined, margin) in enumerate(
        [
            ("c1", [5.0, 0.0, 0.0, 0.0, 0.0, 0.0], "phage", 5.0),
            ("c1", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "unknown", 0.0),
        ]
    ):
        row = {
            "contig_id": cid,
            "window_idx": i,
            "margin": margin,
            "top_logit": scores[0],
            "refined_prediction": refined,
        }
        for sc, val in zip(rf.SCORE_COLS, scores):
            row[sc] = val
        rows.append(row)

    df = pl.DataFrame(rows)
    contig = rf.aggregate_contig(df, mode="gated", min_windows=1)
    assert contig["n_windows_used"][0] == 1
    assert contig["contig_call"][0] == "phage"


def test_aggregate_contig_weighted_uses_margin():
    rows = []
    for i, (cid, scores, refined, margin) in enumerate(
        [
            ("c1", [4.0, 0.0, 0.0, 0.0, 0.0, 0.0], "phage", 4.0),
            ("c1", [0.0, 2.0, 0.0, 0.0, 0.0, 0.0], "virus", 2.0),
        ]
    ):
        row = {
            "contig_id": cid,
            "window_idx": i,
            "margin": margin,
            "top_logit": scores[0],
            "refined_prediction": refined,
        }
        for sc, val in zip(rf.SCORE_COLS, scores):
            row[sc] = val
        rows.append(row)

    df = pl.DataFrame(rows)
    contig = rf.aggregate_contig(df, mode="weighted", min_windows=1)
    # Phage gets 4x weight, virus 2x -> phage still wins.
    assert contig["contig_call"][0] == "phage"
    assert contig["total_weight"][0] == pytest.approx(6.0)


def test_aggregate_contig_merge_split_half():
    rows = [
        {
            "contig_id": "c1",
            "window_idx": 0,
            "margin": 0.5,
            "top_logit": 1.0,
            **dict(zip(rf.SCORE_COLS, [0.0, 0.0, 0.0, 2.0, 1.5, 0.0])),
            "refined_prediction": "bacteria_or_plasmid",
        }
    ]
    df = pl.DataFrame(rows)
    contig = rf.aggregate_contig(df, mode="gated", min_windows=1, merge_split="half")
    # Bacteria should get 2.0 * 0.5 = 1.0; plasmid 1.5 * 0.5 = 0.75.
    assert contig["contig_call"][0] == "bacteria"


def test_aggregate_contig_merge_split_full():
    rows = [
        {
            "contig_id": "c1",
            "window_idx": 0,
            "margin": 0.5,
            "top_logit": 1.0,
            **dict(zip(rf.SCORE_COLS, [0.0, 0.0, 0.0, 1.0, 1.2, 0.0])),
            "refined_prediction": "bacteria_or_plasmid",
        }
    ]
    df = pl.DataFrame(rows)
    contig = rf.aggregate_contig(df, mode="gated", min_windows=1, merge_split="full")
    # Full split: plasmid 1.2 > bacteria 1.0.
    assert contig["contig_call"][0] == "plasmid"


def test_save_and_load_refinement(tmp_path: Path):
    taus = {
        "phage": {"logit": 1.5, "margin": 1.0, "n": 100},
        "virus": {"logit": 0.5, "margin": 0.2, "n": 50},
    }
    path = tmp_path / "refine.yaml"
    rf.save_refinement(
        taus,
        path,
        jaeger_model="test_model",
        quantile=0.05,
        notes="test",
    )
    loaded = rf.load_refinement(path, expect_model="test_model")
    assert loaded["jaeger_model"] == "test_model"
    assert loaded["schema_version"] == 1
    assert loaded["taus"]["phage"]["logit"] == pytest.approx(1.5)


def test_load_refinement_rejects_wrong_model(tmp_path: Path):
    taus = {"phage": {"logit": 0.0, "margin": 0.0, "n": 0}}
    path = tmp_path / "refine.yaml"
    rf.save_refinement(taus, path, jaeger_model="model_a", quantile=0.05)
    with pytest.raises(ValueError):
        rf.load_refinement(path, expect_model="model_b")
