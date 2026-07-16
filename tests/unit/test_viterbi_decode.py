from __future__ import annotations

import numpy as np
import pytest

from jaeger.postprocess.helpers import (
    build_transition_costs,
    default_transition_prior,
    viterbi_decode,
)

CLASS_NAMES = ["bacteria", "phage", "eukarya", "archaea", "plasmid", "virus"]


def _logits(
    dominant: list[int], margin: float = 10.0, n_classes: int = 3
) -> np.ndarray:
    """One-hot-ish logits: ``margin`` at the dominant class, 0 elsewhere."""
    z = np.zeros((len(dominant), n_classes))
    z[np.arange(len(dominant)), dominant] = margin
    return z


# --- viterbi_decode: uniform smoothing ---------------------------------------


def test_zero_switch_cost_equals_independent_argmax():
    rng = np.random.default_rng(42)
    logits = rng.normal(size=(25, 6))
    decoded = viterbi_decode(logits, switch_cost=0.0)
    assert np.array_equal(decoded, np.argmax(logits, axis=-1))


def test_singleton_flip_suppressed():
    # One window favors class 1 with a log-softmax gain of ~10; interior flip
    # costs 2 * lambda, so lambda=6 (cost 12) must suppress it.
    dominant = [0, 0, 0, 1, 0, 0, 0]
    decoded = viterbi_decode(_logits(dominant), switch_cost=6.0)
    assert np.array_equal(decoded, np.zeros(7, dtype=int))


def test_singleton_flip_survives_below_threshold():
    # Same flip, but lambda=4 -> cost 8 < gain 10 -> flip survives.
    dominant = [0, 0, 0, 1, 0, 0, 0]
    decoded = viterbi_decode(_logits(dominant), switch_cost=4.0)
    assert np.array_equal(decoded, np.array(dominant))


def test_sustained_run_preserved():
    # Three consecutive class-1 windows: cumulative gain ~30 > 2 * lambda.
    dominant = [0, 0, 1, 1, 1, 0, 0]
    decoded = viterbi_decode(_logits(dominant), switch_cost=6.0)
    assert np.array_equal(decoded, np.array(dominant))


def test_single_window_contig():
    logits = np.array([[1.0, 5.0, 2.0]])
    decoded = viterbi_decode(logits, switch_cost=2.0)
    assert np.array_equal(decoded, np.array([1]))


def test_output_shape_and_dtype():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(13, 4))
    decoded = viterbi_decode(logits, switch_cost=2.0)
    assert decoded.shape == (13,)
    assert np.issubdtype(decoded.dtype, np.integer)


# --- Binary-head path (stacked [0, z] two-class emissions) -------------------


def _decode_binary(z: np.ndarray, switch_cost: float) -> np.ndarray:
    """Mirror of the binary branch in pred_to_dict."""
    stacked = np.concatenate([np.zeros_like(z), z], axis=-1)
    return viterbi_decode(stacked, switch_cost)


def test_binary_zero_cost_matches_sigmoid_threshold():
    z = np.array([[-1.5], [-1.5], [1.5], [-1.5], [-1.5]])
    decoded = _decode_binary(z, switch_cost=0.0)
    assert np.array_equal(decoded, np.array([0, 0, 1, 0, 0]))


def test_binary_singleton_flip_suppressed():
    # logit 1.5 -> log-odds gain 1.5 < 2 * lambda = 4 -> suppressed.
    z = np.array([[-1.5], [-1.5], [1.5], [-1.5], [-1.5]])
    decoded = _decode_binary(z, switch_cost=2.0)
    assert np.array_equal(decoded, np.zeros(5, dtype=int))


# --- Transition priors --------------------------------------------------------


def test_default_prior_tiers():
    p = default_transition_prior(CLASS_NAMES)
    idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    assert p[idx["bacteria"], idx["phage"]] == pytest.approx(0.5)
    assert p[idx["archaea"], idx["plasmid"]] == pytest.approx(0.5)
    assert p[idx["eukarya"], idx["virus"]] == pytest.approx(0.5)
    assert p[idx["eukarya"], idx["phage"]] == pytest.approx(3.0)
    assert p[idx["bacteria"], idx["eukarya"]] == pytest.approx(3.0)
    assert p[idx["bacteria"], idx["archaea"]] == pytest.approx(3.0)
    assert p[idx["bacteria"], idx["virus"]] == pytest.approx(1.0)
    assert np.allclose(np.diag(p), 0.0)
    assert np.allclose(p, p.T)


def test_prior_maps_by_class_name_not_position():
    # Permuted class order must give the same biological costs.
    permuted = ["virus", "eukarya", "phage"]
    p = default_transition_prior(permuted)
    idx = {name: i for i, name in enumerate(permuted)}
    assert p[idx["virus"], idx["eukarya"]] == pytest.approx(0.5)
    assert p[idx["eukarya"], idx["phage"]] == pytest.approx(3.0)
    assert p[idx["virus"], idx["phage"]] == pytest.approx(1.0)


def test_prior_missing_classes_degrades_to_uniform():
    p = default_transition_prior(["nonphage", "phage"])
    assert np.allclose(p, np.array([[0.0, 1.0], [1.0, 0.0]]))


def test_build_transition_costs_scales_by_lambda():
    costs = build_transition_costs(CLASS_NAMES, switch_cost=2.0)
    idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    assert costs[idx["bacteria"], idx["phage"]] == pytest.approx(1.0)
    assert costs[idx["eukarya"], idx["phage"]] == pytest.approx(6.0)


def test_build_transition_costs_uniform():
    costs = build_transition_costs(CLASS_NAMES, switch_cost=2.0, prior="uniform")
    off_diag = ~np.eye(len(CLASS_NAMES), dtype=bool)
    assert np.allclose(costs[off_diag], 2.0)
    assert np.allclose(np.diag(costs), 0.0)


def test_build_transition_costs_user_matrix_symmetric():
    user = {"bacteria": {"phage": 0.1}}
    costs = build_transition_costs(CLASS_NAMES, switch_cost=2.0, user_matrix=user)
    idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    assert costs[idx["bacteria"], idx["phage"]] == pytest.approx(0.2)
    assert costs[idx["phage"], idx["bacteria"]] == pytest.approx(0.2)
    # untouched pairs stay neutral
    assert costs[idx["bacteria"], idx["virus"]] == pytest.approx(2.0)


def test_cheap_vs_expensive_boundary_behavior():
    # A singleton phage flip with per-window gain ~8 over the background.
    # bacteria background: boundary cost 2 * (2.0 * 0.5) = 2  < 8 -> flip survives
    # eukarya background:  boundary cost 2 * (2.0 * 3.0) = 12 > 8 -> absorbed
    idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    costs = build_transition_costs(CLASS_NAMES, switch_cost=2.0)

    def island(background: str) -> np.ndarray:
        dominant = [idx[background]] * 8
        dominant[4] = idx["phage"]
        return _logits(dominant, margin=8.0, n_classes=len(CLASS_NAMES))

    decoded_bacteria = viterbi_decode(island("bacteria"), transition_costs=costs)
    assert decoded_bacteria[4] == idx["phage"]

    decoded_eukarya = viterbi_decode(island("eukarya"), transition_costs=costs)
    assert np.all(decoded_eukarya == idx["eukarya"])
