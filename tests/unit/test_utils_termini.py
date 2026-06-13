"""Tests for jaeger.utils.termini."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from jaeger.utils import termini


PARASAIL_AVAILABLE = importlib.util.find_spec("parasail") is not None


@pytest.mark.skipif(not PARASAIL_AVAILABLE, reason="parasail not installed")
class TestTerminalRepeats:
    def test_get_alignment_summary(self, tmp_path: Path):
        class MockTraceback:
            query = "AAAAAA"
            ref = "AAAAAA"
            comp = "||||||"

        class MockResult:
            score = 10
            query_start = 0
            query_end = 5
            ref_start = 0
            ref_end = 5
            end_query = 5
            end_ref = 5
            saturated = False
            traceback = MockTraceback()

        summary = termini.get_alignment_summary(
            MockResult(), seq_len=100, record_id="rec1", input_length=100, type_="DTR"
        )
        assert summary["contig_id"] == "rec1"
        assert summary["score"] == 10
