"""Post-hoc refinement of Jaeger window-level and contig-level predictions.

The refinement layer operates in two stages:

1. Window-level: per-class abstain (reject) + targeted merge rules for
   biologically ambiguous pairs (bacteria/plasmid, phage/virus).
2. Contig-level: gated/weighted/unweighted aggregation of refined windows,
   with special handling for merged-label windows.

Thresholds are fit once on a labeled validation set and persisted as YAML.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

SCORE_COLS = [
    "phage_score",
    "virus_score",
    "archaea_score",
    "bacteria_score",
    "plasmid_score",
    "eukarya_score",
]
CLASSES = [c.replace("_score", "") for c in SCORE_COLS]

# Map hedged labels to the strict classes they represent.
MERGE_MAP: dict[str, tuple[str, str]] = {
    "bacteria_or_plasmid": ("bacteria", "plasmid"),
    "virus_any": ("phage", "virus"),
}


def add_score_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add window-level decision features from a logit matrix.

    *df* is expected to contain the six score columns in SCORE_COLS.
    Returns *df* with new columns:
        top_logit, second_logit, margin, top_prob, entropy,
        top_class, second_class.
    """
    S = df.select(SCORE_COLS).to_numpy()
    P = np.exp(S - S.max(axis=1, keepdims=True))
    P = P / P.sum(axis=1, keepdims=True)

    top2 = np.sort(S, axis=1)[:, -2:]
    top_logit = top2[:, 1]
    second_logit = top2[:, 0]
    margin = top_logit - second_logit
    top_prob = P.max(axis=1)
    entropy = -(P * np.log(P + 1e-12)).sum(axis=1)

    top_idx = S.argmax(axis=1)
    second_idx = np.argsort(S, axis=1)[:, -2]

    return df.with_columns(
        [
            pl.Series("top_logit", top_logit),
            pl.Series("second_logit", second_logit),
            pl.Series("margin", margin),
            pl.Series("top_prob", top_prob),
            pl.Series("entropy", entropy),
            pl.Series("top_class", [CLASSES[i] for i in top_idx]),
            pl.Series("second_class", [CLASSES[i] for i in second_idx]),
        ]
    )


def fit_thresholds(
    val_df: pl.DataFrame, quantile: float = 0.05
) -> dict[str, dict[str, float]]:
    """Fit per-class logit/margin thresholds from the correct-diagonal subset.

    For each predicted class *k*, only windows where prediction == true == k are
    used. The threshold is the *quantile* percentile of that correct distribution.
    """
    taus: dict[str, dict[str, float]] = {}
    for k in CLASSES:
        correct = val_df.filter((pl.col("prediction") == k) & (pl.col("true") == k))
        if len(correct) < 30:
            taus[k] = {"logit": -np.inf, "margin": -np.inf, "n": len(correct)}
            continue
        taus[k] = {
            "logit": float(np.quantile(correct["top_logit"].to_numpy(), quantile)),
            "margin": float(np.quantile(correct["margin"].to_numpy(), quantile)),
            "n": len(correct),
        }
    return taus


def refine(
    df: pl.DataFrame,
    taus: dict[str, dict[str, float]],
    merge_bp: bool = True,
    merge_pv: bool = True,
) -> pl.DataFrame:
    """Apply merge rules first, then per-class abstain.

    Returns *df* with a new column ``refined_prediction`` taking values in
    CLASSES ∪ {"unknown", "bacteria_or_plasmid", "virus_any"}.
    """
    top_class = df["top_class"].to_numpy()
    second_class = df["second_class"].to_numpy()
    top_logit = df["top_logit"].to_numpy()
    margin = df["margin"].to_numpy()

    tau_logit = np.array([taus[c]["logit"] for c in top_class])
    tau_margin = np.array([taus[c]["margin"] for c in top_class])

    refined = top_class.copy()

    if merge_bp:
        m = (
            ((top_class == "bacteria") & (second_class == "plasmid"))
            | ((top_class == "plasmid") & (second_class == "bacteria"))
        ) & (margin < tau_margin)
        refined[m] = "bacteria_or_plasmid"

    if merge_pv:
        m = (
            ((top_class == "phage") & (second_class == "virus"))
            | ((top_class == "virus") & (second_class == "phage"))
        ) & (margin < tau_margin)
        refined[m] = "virus_any"

    abstain = ((top_logit < tau_logit) | (margin < tau_margin)) & ~np.isin(
        refined, list(MERGE_MAP.keys())
    )
    refined[abstain] = "unknown"

    return df.with_columns(pl.Series("refined_prediction", refined))


def aggregate_contig(
    window_df: pl.DataFrame,
    mode: str = "gated",
    min_windows: int = 3,
    merge_split: str = "half",
    allow_merged_contig_call: bool = False,
    contig_hedge_margin: float = 1.0,
) -> pl.DataFrame:
    """Aggregate refined per-window predictions into per-contig calls.

    Parameters
    ----------
    window_df
        DataFrame with columns contig_id, refined_prediction, margin, top_logit,
        and the six score columns in SCORE_COLS.
    mode
        ``gated`` (drop unknown windows, unweighted sum), ``weighted`` (drop
        unknown, weight by margin), or ``unweighted`` (baseline: all windows).
    min_windows
        Minimum number of informative windows required to emit a contig call.
    merge_split
        ``half`` gives 0.5 weight to each constituent class for a merged-label
        window; ``full`` gives 1.0.
    allow_merged_contig_call
        If True, contigs whose top-2 classes form a merge pair and whose
        aggregated margin is below ``contig_hedge_margin`` are assigned the
        merged label.
    contig_hedge_margin
        Margin threshold for the contig-level hedge.
    """
    df = window_df

    if mode in ("gated", "weighted"):
        df = df.filter(pl.col("refined_prediction") != "unknown")

    if mode == "weighted":
        base_weight = pl.col("margin").clip(0.0, None)
    else:
        base_weight = pl.lit(1.0)
    df = df.with_columns(base_weight.alias("_w"))

    merge_share = 0.5 if merge_split == "half" else 1.0

    def class_multiplier_expr(class_name: str) -> pl.Expr:
        merged_labels_with_class = [
            lbl for lbl, members in MERGE_MAP.items() if class_name in members
        ]
        return (
            pl.when(pl.col("refined_prediction").is_in(list(MERGE_MAP.keys())))
            .then(
                pl.when(pl.col("refined_prediction").is_in(merged_labels_with_class))
                .then(pl.lit(merge_share))
                .otherwise(pl.lit(0.0))
            )
            .otherwise(pl.lit(1.0))
        )

    agg_exprs = [
        (pl.col(score_col) * pl.col("_w") * class_multiplier_expr(class_name))
        .sum()
        .alias(score_col)
        for score_col, class_name in zip(SCORE_COLS, CLASSES)
    ]
    agg_exprs += [
        pl.len().alias("n_windows_used"),
        pl.col("_w").sum().alias("total_weight"),
        pl.col("refined_prediction")
        .is_in(list(MERGE_MAP.keys()))
        .sum()
        .alias("n_merged_windows"),
    ]

    contig = df.group_by("contig_id").agg(agg_exprs)
    contig = contig.filter(pl.col("n_windows_used") >= min_windows)

    S = contig.select(SCORE_COLS).to_numpy()
    sorted_idx = np.argsort(S, axis=1)
    top_idx = sorted_idx[:, -1]
    second_idx = sorted_idx[:, -2]
    top_val = S[np.arange(len(S)), top_idx]
    second_val = S[np.arange(len(S)), second_idx]

    top_class = np.array([CLASSES[i] for i in top_idx])
    second_class = np.array([CLASSES[i] for i in second_idx])
    contig_margin = top_val - second_val

    if allow_merged_contig_call:
        merge_pairs = {frozenset(members): lbl for lbl, members in MERGE_MAP.items()}
        contig_call = np.array(
            [
                merge_pairs.get(frozenset((t, s)), t)
                if m < contig_hedge_margin and frozenset((t, s)) in merge_pairs
                else t
                for t, s, m in zip(top_class, second_class, contig_margin)
            ]
        )
    else:
        contig_call = top_class

    return contig.with_columns(
        [
            pl.Series("contig_call", contig_call),
            pl.Series("contig_top_class", top_class),
            pl.Series("contig_second_class", second_class),
            pl.Series("contig_top_logit", top_val),
            pl.Series("contig_margin", contig_margin),
        ]
    )


def save_refinement(
    taus: dict[str, dict[str, float]],
    path: str | Path,
    *,
    jaeger_model: str,
    quantile: float,
    merge_rules: tuple[str, ...] = ("bacteria_or_plasmid", "virus_any"),
    val_cohort: str | None = None,
    notes: str | None = None,
) -> None:
    """Persist per-class refinement thresholds + metadata as YAML."""
    payload: dict[str, Any] = {
        "schema_version": 1,
        "jaeger_model": jaeger_model,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quantile": quantile,
        "classes": CLASSES,
        "score_cols": SCORE_COLS,
        "merge_rules": list(merge_rules),
        "val_cohort": val_cohort,
        "notes": notes,
        "taus": {
            k: {
                "logit": float(v["logit"]),
                "margin": float(v["margin"]),
                "n": int(v["n"]),
            }
            for k, v in taus.items()
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False))


def load_refinement(
    path: str | Path, expect_model: str | None = None
) -> dict[str, Any]:
    """Load a refinement calibration file and validate model/schema."""
    meta = yaml.safe_load(Path(path).read_text())
    if meta.get("schema_version", 0) != 1:
        raise ValueError(
            f"Unsupported refinement schema version: {meta.get('schema_version')}"
        )
    if expect_model is not None and meta["jaeger_model"] != expect_model:
        raise ValueError(
            f"Refinement file was calibrated for {meta['jaeger_model']}, "
            f"but current model is {expect_model}. Recalibrate before using."
        )
    return meta
