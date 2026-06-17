"""Tests for jaeger.cli entry points."""

from __future__ import annotations

from click.testing import CliRunner

from jaeger import cli


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--version"])
    assert result.exit_code == 0
    assert "jaeger" in result.output.lower()


def test_health_help():
    runner = CliRunner()
    result = runner.invoke(cli.main, ["health", "--help"])
    assert result.exit_code == 0
    assert "health" in result.output.lower()


def test_predict_help():
    runner = CliRunner()
    result = runner.invoke(cli.main, ["predict", "--help"])
    assert result.exit_code == 0
    assert "input" in result.output.lower()


def test_train_help():
    runner = CliRunner()
    result = runner.invoke(cli.main, ["train", "--help"])
    assert result.exit_code == 0


def test_utils_help():
    runner = CliRunner()
    result = runner.invoke(cli.main, ["utils", "--help"])
    assert result.exit_code == 0
    assert "combine" in result.output.lower()


def test_taxonomy_help():
    runner = CliRunner()
    result = runner.invoke(cli.main, ["taxonomy", "--help"])
    assert result.exit_code == 0


def test_download_help():
    runner = CliRunner()
    result = runner.invoke(cli.main, ["download", "--help"])
    assert result.exit_code == 0


def test_optimize_data_help_includes_new_flags():
    runner = CliRunner()
    result = runner.invoke(cli.main, ["utils", "optimize-data", "--help"])
    assert result.exit_code == 0
    assert "--max-memory-mb" in result.output
    assert "--pad" in result.output
