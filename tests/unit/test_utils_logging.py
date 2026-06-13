"""Tests for jaeger.utils.logging."""

from __future__ import annotations

from pathlib import Path

from jaeger.utils import logging as jaeger_logging


def test_description_contains_banner():
    text = jaeger_logging.description("1.0.0")
    assert "Jaeger" in text
    assert "1.0.0" in text


def test_get_logger(tmp_path: Path):
    logger = jaeger_logging.get_logger(tmp_path, "test.log", level=1)
    assert logger.name == "jaeger"
    assert logger.isEnabledFor(20)  # INFO level
    assert list(tmp_path.glob("*test.log"))
