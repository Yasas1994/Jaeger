"""Tests for jaeger.utils.misc."""

from __future__ import annotations

import json
import tempfile
from decimal import Decimal
from pathlib import Path

import pytest

from jaeger.utils import misc


class TestSafeDivide:
    def test_safe_divide_normal(self):
        assert misc.safe_divide(10, 2) == 5.0

    def test_safe_divide_zero(self):
        assert misc.safe_divide(10, 0) == 0.0


class TestTimeFormatting:
    def test_format_seconds_minutes(self):
        assert "minutes" in misc.format_seconds(125)

    def test_format_seconds_seconds(self):
        assert misc.format_seconds(45).endswith("seconds")


class TestRounding:
    def test_round_num(self):
        assert misc.round_num(3.14159, 2) == Decimal("3.14")

    def test_numerize(self):
        assert "K" in misc.numerize(1500)
        assert misc.numerize(999) == "999"


class TestSignalIterators:
    def test_signal_fl(self):
        values = list(misc.signal_fl([1, 2, 3]))
        assert values == [(True, 1), (0, 2), (1, 3)]

    def test_signal_l(self):
        values = list(misc.signal_l([1, 2, 3]))
        assert values == [(0, 1), (0, 2), (1, 3)]


class TestJsonHelpers:
    def test_json_to_dict(self, tmp_path: Path):
        path = tmp_path / "data.json"
        path.write_text(json.dumps({"a": 1}))
        assert misc.json_to_dict(path) == {"a": 1}

    def test_add_data_to_json(self, tmp_path: Path):
        path = tmp_path / "data.json"
        path.write_text(json.dumps({"items": []}))
        misc.add_data_to_json(path, {"x": 2}, list_key="items")
        data = json.loads(path.read_text())
        assert {"x": 2} in data["items"]


class TestAvailableModels:
    def test_scan_model_directory(self, tmp_path: Path):
        model_dir = tmp_path / "experiment" / "model"
        model_dir.mkdir(parents=True)
        graph_dir = model_dir / "test_model_graph"
        graph_dir.mkdir()
        (model_dir / "test_model_classes.yaml").write_text("classes:\n")
        (model_dir / "test_model_project.yaml").write_text("project:\n")

        am = misc.AvailableModels(tmp_path / "experiment")
        assert "test_model" in am.info
        assert am.info["test_model"]["graph"] == graph_dir
        assert "classes" in am.info["test_model"]
        assert "project" in am.info["test_model"]

    def test_empty_directory(self, tmp_path: Path):
        am = misc.AvailableModels(tmp_path)
        assert am.info == {}


class TestGetModelId:
    def test_get_model_id(self):
        assert misc.get_model_id("jaeger_500bp_test_fragment") == "500bp_test"
